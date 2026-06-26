"""
Unified Multimodal PDF Reader & Dashboard
====================================================================
Architecture (HYBRID FUSION TRACKING + PDF READING):
- Analyzers: Gaze (MediaPipe/ResNet), Emotion (MediaPipe), Epistemic (MediaPipe), Mouse
- Gaze features: RAW output (No Kalman, no temporal smoothing, no deadzones)
- PDF features: PyMuPDF rendering, FUSED Multimodal Stuck Detection, Gemini Summarization
"""

import cv2
import time
import numpy as np
import threading
import argparse
import sys
from collections import deque
from pathlib import Path

# External Modalities (Assumed existing)
from local_epistemic_tracker import LocalEpistemicTracker
from emotion_wheel import EmotionDetector, PlutchikWheel
from mouse_analyzer import MouseReadingAnalyzer
from gaze_analyzer import GazeReadingAnalyzer
from gaze_core import GazeReader, CALIB_PTS, DRIFT_PTS

# Newly Modularized Files
from ui_components import (SCREEN_W, SCREEN_H, SANDBOX_W, SANDBOX_H, WHEEL_W, WHEEL_H,
                           SAMPLES_NEEDED, BLINK_BLENDSHAPE_THRESHOLD, SCROLL_STEP,
                           draw_target_dot, GazeHeatmap, draw_dialog)
from pdf_reader import PDFDocument, summarize
from dialog_controller import DialogController
from face_modality import FaceModalityTracker


class ReaderHelperApp:
    def __init__(self, pdf_path, zoom, dwell):
        self.SCREEN_W, self.SCREEN_H = SCREEN_W, SCREEN_H
        self.SANDBOX_W, self.SANDBOX_H = SANDBOX_W, SANDBOX_H
        self.WHEEL_W, self.WHEEL_H = WHEEL_W, WHEEL_H
        
        # Initialize PDF Document
        self.pdf = PDFDocument(pdf_path, zoom)
        self.page = 0
        self.scroll_y = 0
        self.max_scroll = 0
        self.hov_buf = deque(maxlen=20)
        self.stable_hov = None
        
        self.dlg_active = False
        self.dlg_para = None
        self.dlg_summary = {'text': None}
        
        # Initialize Core Trackers
        self.gaze_reader = GazeReader(screen_w=SCREEN_W, screen_h=SCREEN_H, cam_src=0).start()
        self.face_tracker = FaceModalityTracker(blink_threshold=BLINK_BLENDSHAPE_THRESHOLD)
        self.heatmap = GazeHeatmap()
        self.emotion_detector = EmotionDetector()
        self.emotion_wheel = PlutchikWheel(width=self.WHEEL_W, height=self.WHEEL_H)
        self.epistemic_tracker = LocalEpistemicTracker(window_frames=90, min_frames=20)
        self.mouse_tracker = MouseReadingAnalyzer()
        self.gaze_tracker = GazeReadingAnalyzer(window_size=5.0)
        
        # FUSED Dialog Controller
        self.dialog_controller = DialogController(dwell=dwell)
        self.max_score = None
        # State tracking
        self.inference_mode = False
        self.current_gaze_score = 0.0
        self.current_gaze_state = "Initializing"
        self.calib_index = 0
        self.sampling_active = False
        
        # Drift Tracking
        self.drift_mode = False
        self.drift_index = 0

        # Unfiltered Raw Coordinate tracking
        self.gaze_x, self.gaze_y = self.SCREEN_W // 2, self.SCREEN_H // 2
        self.system_action = None

        self.mouse_tracker.start()
        self.last_frame_id = -1
        self.display_image = None
        self.running = True

    def run(self):
        print("Starting Unified Multimodal PDF Reader...")
        time.sleep(1.0)
        cv2.namedWindow('Reader & Dashboard', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Reader & Dashboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while self.running:
            self._process_camera_feed()
            self._update_and_fuse()
            self._render()
            self._handle_input()
            
        self.cleanup()

    def _process_camera_feed(self):
        self.gaze_reader.update()
        raw_frame = self.gaze_reader._stream.read()
        current_frame_id = self.gaze_reader._stream.fid
        
        if raw_frame is not None and current_frame_id > self.last_frame_id:
            self.display_image = cv2.flip(raw_frame, 1)
            image_rgb = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
            self.face_tracker.process_frame(image_rgb, int(time.time() * 1000))
            self.last_frame_id = current_frame_id

    def _update_and_fuse(self):
        if self.display_image is None: return

        # Extract Modality States
        process_new_frame, bs = self.face_tracker.update_state()
        mouse_data = self.mouse_tracker.get_data()
        self.mouse_score = mouse_data["score"] if mouse_data else None # Keep this if you still draw it on the HUD
        emotion_state = self.emotion_detector.update(bs) if process_new_frame else self.emotion_detector.scores
        
        epistemic_state = {}
        if process_new_frame:
            self.epistemic_tracker.update(bs)
            epistemic_state = getattr(self.epistemic_tracker, 'current_state', {}) 
            
        if self.face_tracker.get_fatigue_score() > 0.15: 
            self.dialog_controller.system_message = "INTERVENTION: High Fatigue Detected. Consider a screen break."
            self.dialog_controller.help_active = True

        # Process RAW Gaze Data Coords First
        if self.gaze_reader.is_calibrated:
            raw_norm_x, raw_norm_y = self.gaze_reader.get_gaze_norm()
            self.gaze_x = int(raw_norm_x * self.SCREEN_W)
            self.gaze_y = int(raw_norm_y * self.SCREEN_H)
            self.heatmap.update(self.gaze_x, self.gaze_y)

        # Collision detection (Requires coords to check if on paper)
        self.stable_hov = None
        hov = None
        if self.inference_mode and self.gaze_reader.is_calibrated:
            pg_img, paras = self.pdf.get_page(self.page)
            ph, pw = pg_img.shape[:2]
            ox = max(0, (self.SCREEN_W - pw) // 2)
            gpx = self.gaze_x - ox
            gpy = self.gaze_y + self.scroll_y
            
            if not self.dlg_active and paras and pw > 0:
                ax0 = min(p["bbox"][0] for p in paras)
                ax1 = max(p["bbox"][2] for p in paras)
                hm  = (ax1 - ax0) * 0.35
                if ax0 - hm <= gpx <= ax1 + hm:
                    bd, bi = float('inf'), None
                    for pi, para in enumerate(paras):
                        bx0, by0, bx1, by1 = para["bbox"]
                        vd = 0 if by0 <= gpy <= by1 else min(abs(gpy - by0), abs(gpy - by1))
                        if vd < 60 and vd < bd: bd = vd; bi = pi
                    hov = bi

            self.hov_buf.append(hov)
            counts = {}
            for v in self.hov_buf:
                if v is not None: counts[v] = counts.get(v, 0) + 1
            nones = self.hov_buf.count(None)
            self.stable_hov = (max(counts, key=counts.get) if counts and nones < len(self.hov_buf) // 2 else None)

        # Now Extract Gaze Score & State
        if self.gaze_reader.is_calibrated:
            is_on_paper = (hov is not None) if self.inference_mode else True
            gaze_data = self.gaze_tracker.process_point(self.gaze_x, self.gaze_y, on_paper=is_on_paper)
            self.current_gaze_score = gaze_data["score"]
            self.current_gaze_state = gaze_data["state"]
        else:
            self.current_gaze_score = 0.0
            self.current_gaze_state = "Initializing"

        # Fused Evaluation passing the states directly to the controller
        # Pack gaze score and state back into a dict for the controller
        gaze_data = {"score": self.current_gaze_score, "state": self.current_gaze_state}

        self.system_action = self.dialog_controller.evaluate(
            mouse_data, gaze_data, epistemic_state, emotion_state, self.stable_hov
        )
        # Trigger Dialog Generation ONLY if Dialog Controller explicitly flags `needs_summary`
        triggered_para_idx = self.system_action.get("triggered_para")
        needs_summary = self.system_action.get("needs_summary", False)
        
        if triggered_para_idx is not None and needs_summary and not self.dlg_active and self.inference_mode:
            _, paras = self.pdf.get_page(self.page)
            if paras:
                self.dlg_active = True
                self.dlg_para = paras[triggered_para_idx]
                self.dlg_summary['text'] = None
                
                def _summ(text, box): box['text'] = summarize(text)
                threading.Thread(target=_summ, args=(self.dlg_para["text"], self.dlg_summary), daemon=True).start()

    def _render(self):
        canvas = np.zeros((self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8)

        # Dashboard / Calibration Mode
        if not self.gaze_reader.is_calibrated or not self.inference_mode:
            if not self.gaze_reader.is_calibrated:
                canvas[:] = 18
                self._draw_calibration_ui(canvas, self.SCREEN_W, self.SCREEN_H)
            else:
                sandbox = np.ones((self.SANDBOX_H, self.SANDBOX_W, 3), dtype=np.uint8) * 18
                wheel_canvas = self.emotion_wheel.render(self.emotion_detector.scores)
                epistemic_canvas = np.zeros((self.WHEEL_H, self.WHEEL_W, 3), dtype=np.uint8)
                self.epistemic_tracker.render(epistemic_canvas)

                canvas[0:self.SANDBOX_H, 0:self.SANDBOX_W] = sandbox                     
                canvas[0:self.WHEEL_H, self.SANDBOX_W:self.SCREEN_W] = wheel_canvas               
                canvas[self.WHEEL_H:self.SCREEN_H, self.SANDBOX_W:self.SCREEN_W] = epistemic_canvas        
                
                self._draw_tracking_ui(canvas, self.SCREEN_W, self.SCREEN_H)
                self._draw_fusion_hud(canvas, self.mouse_score, self.system_action)
                cv2.putText(canvas, "Press [I] to enter PDF READER MODE", (1320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Inference Mode (PDF Reader)
        else:
            pg_img, paras = self.pdf.get_page(self.page)
            ph, pw = pg_img.shape[:2]
            self.max_scroll = max(0, ph - self.SCREEN_H)
            self.scroll_y = min(self.scroll_y, self.max_scroll)
            
            ox = max(0, (self.SCREEN_W - pw) // 2)
            xe = min(ox + pw, self.SCREEN_W)
            sh = min(self.SCREEN_H, ph - self.scroll_y)
            
            canvas[:] = 30
            canvas[0:sh, ox:xe] = pg_img[self.scroll_y:self.scroll_y + sh, :xe - ox]

            # Draw highlight for stable hovered paragraph
            for pi, para in enumerate(paras):
                if pi == self.stable_hov:
                    bx0, by0, bx1, by1 = para["bbox"]
                    sy0 = by0 - self.scroll_y; sy1 = by1 - self.scroll_y
                    if sy1 > 0 and sy0 < self.SCREEN_H:
                        cv2.rectangle(canvas, (ox+bx0, int(sy0)), (ox+bx1, int(sy1)), (255, 220, 50), 2)

            # Draw Gaze Cursor
            if not self.dlg_active:
                cv2.line(canvas, (self.gaze_x - 14, self.gaze_y), (self.gaze_x + 14, self.gaze_y), (0, 255, 180), 1)
                cv2.line(canvas, (self.gaze_x, self.gaze_y - 14), (self.gaze_x, self.gaze_y + 14), (0, 255, 180), 1)
                cv2.circle(canvas, (self.gaze_x, self.gaze_y), 3, (0, 255, 180), -1)

            # Drift UI overlay
            if self.drift_mode:
                ddx = int(DRIFT_PTS[self.drift_index][0] * self.SCREEN_W)
                ddy = int(DRIFT_PTS[self.drift_index][1] * self.SCREEN_H)
                cv2.putText(canvas, f"Drift ({self.drift_index+1}/{len(DRIFT_PTS)}) - SPACE", (20, self.SCREEN_H-50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 220, 50), 2)
                
                if self.gaze_reader._is_drift and self.sampling_active:
                    prog = self.gaze_reader.calibration_progress
                    draw_dot(canvas, ddx, ddy, prog, f"Hold still... {int(prog * self.gaze_reader.DRIFT_SAMPLES)}/{self.gaze_reader.DRIFT_SAMPLES}")
                    
                    if self.gaze_reader.drift_point_ready:
                        self.gaze_reader.commit_drift_point()
                        self.sampling_active = False
                        self.drift_index += 1
                        if self.drift_index >= len(DRIFT_PTS):
                            self.drift_mode = False
                            self.drift_index = 0
                            self.dialog_controller.reset_dwell()
                else:
                    pp = int(8 + 5 * abs(np.sin(time.time() * 3)))
                    cv2.circle(canvas, (ddx, ddy), pp+5, (255, 220, 50), 2)
                    cv2.circle(canvas, (ddx, ddy), pp, (200, 160, 0), -1)

            scroll_pct = int(100 * self.scroll_y / self.max_scroll) if self.max_scroll > 0 else 0
            cv2.putText(canvas, f"Page {self.page+1}/{self.pdf.n}  [{scroll_pct}%]", (20, self.SCREEN_H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
            cv2.putText(canvas, "[I] Dashboard | j/k=scroll n/p=page d=drift r=recal q=quit", (self.SCREEN_W-650, self.SCREEN_H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            # --- TOP-LEFT STATE TRACKER ---
            tracker_text = f"STATE TRACKER: {self.current_gaze_state}"
            (tw, th), _ = cv2.getTextSize(tracker_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(canvas, (10, 10), (20 + tw + 10, 20 + th + 15), (30, 30, 30), -1)
            cv2.rectangle(canvas, (10, 10), (20 + tw + 10, 20 + th + 15), (0, 255, 255), 1)
            cv2.putText(canvas, tracker_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Draw HUD from Multimodal Dashboard (Shifted down to avoid overlapping the tracker)
            if self.system_action and self.system_action['help_active']:
                cv2.putText(canvas, f"MULTIMODAL SYS: {self.system_action['message']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

            if self.face_tracker.is_blinking:
                cv2.putText(canvas, "BLINK DETECTED", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if self.dlg_active:
                draw_dialog(canvas, self.dlg_summary)

        cv2.imshow('Reader & Dashboard', canvas)
        
    def _draw_calibration_ui(self, canvas, w, h):
        tx = int(CALIB_PTS[self.calib_index][0] * w)
        ty = int(CALIB_PTS[self.calib_index][1] * h)
        
        if self.sampling_active:
            progress = self.gaze_reader.calibration_progress
            draw_target_dot(canvas, tx, ty, progress, f"Hold still... {int(progress*100)}%")
            
            if self.gaze_reader.calibration_point_ready:
                self.gaze_reader.commit_calibration_point()
                self.sampling_active = False
                self.calib_index += 1
                
                if self.calib_index == len(CALIB_PTS):
                    self.gaze_reader.fit()
        else:
            pulse = int(10 + 6 * abs(np.sin(time.time() * 3)))
            cv2.circle(canvas, (tx, ty), pulse + 6, (255, 255, 255), 2)
            cv2.circle(canvas, (tx, ty), pulse, (0, 0, 220), -1)
            
            text = f"Look at dot ({self.calib_index+1}/{len(CALIB_PTS)}) — press SPACE"
            text_w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0][0]
            cv2.putText(canvas, text, ((w - text_w) // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
        
        for i, (px, py) in enumerate(CALIB_PTS):
            cv2.circle(canvas, (int(px * w), int(py * h)), 8 if i < self.calib_index else 5 if i == self.calib_index else 6, (0, 200, 0) if i < self.calib_index else (255, 255, 255) if i == self.calib_index else (80, 80, 80), -1)

    def _draw_tracking_ui(self, canvas, w, h):
        self.heatmap.render(canvas)
        cv2.line(canvas, (self.gaze_x - 14, self.gaze_y), (self.gaze_x + 14, self.gaze_y), (255, 255, 255), 1)
        cv2.line(canvas, (self.gaze_x, self.gaze_y - 14), (self.gaze_x, self.gaze_y + 14), (255, 255, 255), 1)
        cv2.circle(canvas, (self.gaze_x, self.gaze_y), 4, (255, 255, 255), -1)
        
        cv2.putText(canvas, "Tracking Base — 'r' recalibrate  'q' quit", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        status_text = f"RAW Gaze Output: X({self.gaze_x}) Y({self.gaze_y}) | State: {self.current_gaze_state}"
        if self.face_tracker.is_blinking: status_text += " [BLINKING]"
        cv2.putText(canvas, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255) if self.face_tracker.is_blinking else (200,200,200), 1)

    def _draw_fusion_hud(self, canvas, mouse_score, action_state):
        hud_x, hud_y = 30, self.SANDBOX_H - 120
        cv2.rectangle(canvas, (hud_x - 10, hud_y - 40), (hud_x + 550, hud_y + 100), (40, 40, 40), -1)
        
        color = (0, 100, 255) if action_state["help_active"] else (255, 255, 255)
        cv2.putText(canvas, f"SYSTEM: {action_state['message']}", (hud_x, hud_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        mouse_text = f"Mouse PAR Score: {mouse_score:.2f}" if mouse_score is not None else "Mouse PAR Score: INACTIVE"
        mouse_color = (200, 200, 200) if mouse_score is not None else (100, 100, 100)
            
        cv2.putText(canvas, mouse_text, (hud_x, hud_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouse_color, 1)
        cv2.putText(canvas, f"Fusion Struggle Level: {action_state['struggle_level']:.2f}", (hud_x, hud_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    def _handle_input(self):
        key = cv2.waitKey(1)
        key_char = key & 0xFF
        
        if key_char == ord('q'): 
            self.running = False
        elif key_char == ord('i') and self.gaze_reader.is_calibrated:
            self.inference_mode = not self.inference_mode
            
        # Dialog Inputs
        elif key_char in (ord('y'), ord('Y')) and self.dlg_active: 
            self.dlg_active = False; self.dlg_para = None
        elif key_char in (ord('n'), ord('N')) and self.dlg_active:
            self.dlg_active = False; self.dlg_para = None; self.dlg_summary['text'] = None
            
        # Reading Inputs
        elif not self.dlg_active:
            if key_char == ord('j'):
                self.scroll_y = min(self.max_scroll, self.scroll_y + SCROLL_STEP)
            elif key_char == ord('k'):
                self.scroll_y = max(0, self.scroll_y - SCROLL_STEP)
            elif key_char == ord('n') and self.inference_mode:
                self.page = min(self.page + 1, self.pdf.n - 1)
                self.scroll_y = 0; self.dialog_controller.reset_dwell(); self.hov_buf.clear()
            elif key_char == ord('p') and self.inference_mode:
                self.page = max(self.page - 1, 0)
                self.scroll_y = 0; self.dialog_controller.reset_dwell(); self.hov_buf.clear()
            
            # Calibration / Drift Inputs
            elif key_char == ord('r'):
                self.calib_index = 0
                self.sampling_active = False
                self.inference_mode = False 
                self.drift_mode = False
                self.drift_index = 0
                self.gaze_x, self.gaze_y = self.SCREEN_W//2, self.SCREEN_H//2
                self.gaze_reader.reset_calibration()
                self.heatmap.reset()
                self.dialog_controller.reset_dwell(); self.hov_buf.clear()
            elif key_char == ord('d') and self.gaze_reader.is_calibrated and self.inference_mode and not self.drift_mode:
                self.drift_mode = True
                self.drift_index = 0
            elif key_char == ord(' '):
                # Sample Calibration
                if not self.gaze_reader.is_calibrated and not self.sampling_active: 
                    self.sampling_active = True
                    tx, ty = CALIB_PTS[self.calib_index]
                    self.gaze_reader.begin_calibration_point(tx, ty, SAMPLES_NEEDED)
                # Sample Drift
                elif self.gaze_reader.is_calibrated and self.drift_mode and not self.sampling_active:
                    self.sampling_active = True
                    tx, ty = DRIFT_PTS[self.drift_index]
                    self.gaze_reader.begin_drift_point(tx, ty, self.gaze_reader.DRIFT_SAMPLES)

    def cleanup(self):
        print("Cleaning up modules...")
        self.gaze_reader.stop()
        self.face_tracker.stop()
        self.mouse_tracker.stop()
        self.pdf.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Multimodal PDF Reader Dashboard")
    ap.add_argument('--pdf', default=None, help="Path to the PDF document to read.")
    ap.add_argument('--dwell', type=float, default=5.0, help="Dwell time in seconds to trigger stuck dialog.")
    ap.add_argument('--zoom', type=float, default=None, help="PDF render zoom multiplier.")
    args = ap.parse_args()

    if not getattr(args, 'pdf', None) or not Path(args.pdf).exists():
        print("\n[ERROR] You must provide a valid PDF file path!")
        print("Example: python main.py --pdf \"/path/to/your/document.pdf\"\n")
        sys.exit(1)

    app = ReaderHelperApp(
        pdf_path=args.pdf, 
        zoom=args.zoom,
        dwell=args.dwell,
    )
    app.run()
