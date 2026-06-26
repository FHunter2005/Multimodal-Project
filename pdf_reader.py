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
import mediapipe as mp
import threading
import argparse
import os
import sys
from collections import deque
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import fitz
except ImportError:
    print("[ERROR] pip install pymupdf")
    sys.exit(1)

# Assume these are available in your local environment
from local_epistemic_tracker import LocalEpistemicTracker
from emotion_wheel import EmotionDetector, PlutchikWheel
from mouse_analyzer import MouseReadingAnalyzer
from gaze_analyzer import GazeReadingAnalyzer
from gaze_core import GazeReader, CALIB_PTS, DRIFT_PTS

# =============================================================
# 1. CORE CONSTANTS & UI HELPERS
# =============================================================
SCREEN_W, SCREEN_H = 1920, 1080
SANDBOX_W, SANDBOX_H, WHEEL_W, WHEEL_H = 1280, 1080, 640, 540



SAMPLES_NEEDED = 10 
BLINK_BLENDSHAPE_THRESHOLD = 0.35 
SCROLL_STEP = 120

def draw_target_dot(canvas, tx, ty, progress, label):
    cv2.ellipse(canvas, (tx, ty), (22, 22), -90, 0, int(360 * progress), (0, 255, 100), 3)
    cv2.circle(canvas, (tx, ty), 12, (0, 200, 80), -1)
    cv2.putText(canvas, label, ((SCREEN_W - 440) // 2, SCREEN_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

class GazeHeatmap:
    def __init__(self, width=SCREEN_W, height=SCREEN_H, blob_sigma=55.0, decay=0.8, alpha=0.55):
        self.w, self.h, self.decay, self.alpha, self.acc = width, height, decay, alpha, np.zeros((height, width), dtype=np.float32)
        r = int(blob_sigma * 3.5); ax = np.arange(-r, r + 1, dtype=np.float32); xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * blob_sigma**2)); self._kernel, self._r = (kernel / kernel.max()).astype(np.float32), r
        
    def update(self, gx: int, gy: int):
        self.acc *= self.decay
        x0, x1, y0, y1 = max(gx - self._r, 0), min(gx + self._r + 1, self.w), max(gy - self._r, 0), min(gy + self._r + 1, self.h)
        kx0, ky0 = x0 - (gx - self._r), y0 - (gy - self._r)
        self.acc[y0:y1, x0:x1] += self._kernel[ky0:ky0 + (y1 - y0), kx0:kx0 + (x1 - x0)]
        
    def render(self, canvas: np.ndarray) -> np.ndarray:
        peak = self.acc.max()
        if peak < 1e-3: return canvas
        norm = np.clip(self.acc / peak, 0.0, 1.0)
        alpha_map = (norm * self.alpha * (norm > 0.05).astype(np.float32))
        color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        for c in range(3): canvas[:, :, c] = np.clip(color[:, :, c] * alpha_map + canvas[:, :, c] * (1.0 - alpha_map), 0, 255).astype(np.uint8)
        return canvas
        
    def reset(self): self.acc[:] = 0.0

# =============================================================
# 2. PDF & SUMMARIZATION MODULES
# =============================================================
class PDFDocument:
    def __init__(self, path, zoom=None):
        self.doc = fitz.open(path)
        self.n = len(self.doc)
        if zoom is None:
            pw = self.doc[0].rect.width
            self.zoom = (SCREEN_W * 0.95) / pw
        else:
            self.zoom = zoom
        self._cache = {}

    def get_page(self, num):
        if num in self._cache: return self._cache[num]
        page = self.doc[num]
        pix  = page.get_pixmap(matrix=fitz.Matrix(self.zoom, self.zoom), alpha=False)
        img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        img  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ph   = page.rect.height; paras = []
        
        for b in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
            if b["type"] != 0: continue
            x0, y0, x1, y1 = [c * self.zoom for c in b["bbox"]]
            text = " ".join(s["text"] for l in b.get("lines",[]) for s in l.get("spans",[])).strip()
            if len(text) < 80 or b["bbox"][1] < ph*0.12: continue
            lines = b.get("lines", [])
            if len(lines) <= 2:
                avg = np.mean([s.get("size",10) for l in lines for s in l.get("spans",[])] or [10])
                if avg > 13: continue
            paras.append({"bbox": (int(x0), int(y0), int(x1), int(y1)), "text": text})
            
        self._cache[num] = (img, paras)
        return img, paras

    def close(self): self.doc.close()

def summarize(text):
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        try:
            from google import genai
            client = genai.Client(api_key=key)
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Summarize in 2-3 simple sentences for a student struggling with this:\n\n" + text)
            return resp.text.strip()
        except Exception as e: return f"[API error: {e}]"
    return "[Set GEMINI_API_KEY env var for AI summary]\n\n" + text[:400]

def wrap_text(text, n=65):
    words = text.split(); lines = []; line = ""
    for w in words:
        if len(line)+len(w)+1 > n:
            if line: lines.append(line)
            line = w
        else: line = (line+" "+w).strip()
    if line: lines.append(line)
    return lines

def draw_dialog(canvas, dlg_summary):
    dim = canvas.copy(); canvas[:] = canvas // 3
    cv2.addWeighted(dim, 0.15, canvas, 0.85, 0, canvas)
    dw, dh = 900, 500; dx0=(SCREEN_W-dw)//2; dy0=(SCREEN_H-dh)//2
    cv2.rectangle(canvas, (dx0,dy0),    (dx0+dw,dy0+dh), (25,25,35),    -1)
    cv2.rectangle(canvas, (dx0,dy0),    (dx0+dw,dy0+dh), (100,180,255),  2)
    cv2.rectangle(canvas, (dx0,dy0),    (dx0+dw,dy0+52), (40,40,60),    -1)
    cv2.putText(canvas, "Looks like you might be stuck on this paragraph.",
                (dx0+20, dy0+36), cv2.FONT_HERSHEY_DUPLEX, 0.85, (100,210,255), 1, cv2.LINE_AA)
    cv2.line(canvas, (dx0, dy0+52), (dx0+dw, dy0+52), (100,180,255), 1)
    sumtext = dlg_summary['text']
    if sumtext is None:
        cv2.putText(canvas, "Generating AI summary...", (dx0+24, dy0+110),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (160,160,160), 1, cv2.LINE_AA)
    else:
        for i, line in enumerate(wrap_text(sumtext)[:10]):
            cv2.putText(canvas, line, (dx0+24, dy0+96+i*36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.72, (230,230,230), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (dx0+24,   dy0+dh-64), (dx0+220,  dy0+dh-18), (0,160,70),   -1)
    cv2.putText(canvas, "Y   Keep summary",       (dx0+34,   dy0+dh-34), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (dx0+dw-210,dy0+dh-64), (dx0+dw-18,dy0+dh-18), (170,50,50), -1)
    cv2.putText(canvas, "N   Dismiss",            (dx0+dw-200,dy0+dh-34), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

# =============================================================
# 3. DIALOG CONTROLLER (FUSED)
# =============================================================
class DialogController:
    def __init__(self, dwell=5.0, cooldown=20.0):
        self.struggle_frames = 0
        self.help_active = False
        self.system_message = "Listening to User State..."
        
        # Paragraph Stuck Tracking
        self.dwell = dwell
        self.cooldown = cooldown
        self.cur_para = None
        self.para_t0 = None
        self.para_fired = {}
        self.para_fv = False

    def evaluate(self, mouse_score, gaze_score, epistemic_state, emotion_scores, current_para):
        # 1. Base Multimodal Struggle Calculations
        confusion = epistemic_state.get('confusion', 0.0) if isinstance(epistemic_state, dict) else 0.0
        anger = emotion_scores.get('anger', 0.0) if isinstance(emotion_scores, dict) else 0.0
        sadness = emotion_scores.get('sadness', 0.0) if isinstance(emotion_scores, dict) else 0.0
        
        plutchik_frustration = (anger * 0.70) + (sadness * 0.30)
        face_struggle = max(confusion, plutchik_frustration)

        user_is_stuck = False
        if gaze_score > 0.75: user_is_stuck = True
        elif gaze_score > 0.40 and face_struggle > 0.40: user_is_stuck = True
        elif mouse_score is not None:
            if face_struggle > 0.40 and mouse_score > 0.40: user_is_stuck = True
            elif mouse_score > 0.85: user_is_stuck = True
        elif face_struggle > 0.70: user_is_stuck = True

        if user_is_stuck: self.struggle_frames += 1
        else: self.struggle_frames = max(0, self.struggle_frames - 2)

        TRIGGER_THRESHOLD = 300 
        
        # 2. Paragraph Dwell Fusion & Decision Engine
        triggered_para = None
        needs_summary = False
        now = time.time()
        
        if current_para != self.cur_para:
            self.cur_para = current_para
            self.para_t0 = now if current_para is not None else None
            self.para_fv = False
        elif current_para is not None and not self.para_fv:
            
            # Dynamically adjust required reading time based on multimodal struggle
            dynamic_dwell = self.dwell * 0.5 if user_is_stuck else self.dwell
            
            if now - self.para_t0 >= dynamic_dwell and now - self.para_fired.get(current_para, 0) >= self.cooldown:
                self.para_fired[current_para] = now
                self.para_fv = True
                
                # --- DECISION ENGINE: DOES THE USER ACTUALLY NEED A SUMMARY? ---
                if plutchik_frustration > 0.6 and plutchik_frustration > confusion:
                    # User is primarily highly frustrated. A summary dialog blocking the screen might enrage them.
                    # Suggest a break instead.
                    self.help_active = True
                    self.system_message = "INTERVENTION: High Frustration. Take a short break, don't force it."
                else:
                    # User is confused, overloaded, or just naturally reading slowly. Provide a summary.
                    triggered_para = current_para
                    needs_summary = True
                    self.help_active = True
                    self.system_message = "INTERVENTION: Cognitive load detected. Generating paragraph summary..."
        
        # 3. Fallback System Messages (If struggle continues without hovering on one specific paragraph)
        if self.struggle_frames > TRIGGER_THRESHOLD and not self.help_active:
            self.help_active = True
            if plutchik_frustration > confusion: self.system_message = "INTERVENTION: High Frustration. Take a short break."
            elif gaze_score > 0.6: self.system_message = "INTERVENTION: High Cognitive Load detected. Relax your eyes."
            else: self.system_message = "INTERVENTION: Confusion detected. Reading pace adjusted."
                
        elif self.struggle_frames == 0 and self.help_active and not needs_summary:
            self.help_active = False
            self.system_message = "User engaged. Normal reading."

        return {
            "help_active": self.help_active,
            "message": self.system_message,
            "struggle_level": min(1.0, self.struggle_frames / float(TRIGGER_THRESHOLD)),
            "gaze_score": gaze_score,
            "face_score": face_struggle,
            "triggered_para": triggered_para,
            "needs_summary": needs_summary
        }

    def reset_intervention(self):
        self.struggle_frames = 0
        self.help_active = False
        self.reset_dwell()
        
    def reset_dwell(self):
        self.cur_para = None
        self.para_t0 = None
        self.para_fv = False

# =============================================================
# 4. MAIN APPLICATION CLASS
# =============================================================
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
        
        # Initialize Core GazeReader
        self.gaze_reader = GazeReader(screen_w=SCREEN_W, screen_h=SCREEN_H, cam_src=0).start()
        
        # Initialize Analyzers
        self.heatmap = GazeHeatmap()
        self.emotion_detector = EmotionDetector()
        self.emotion_wheel = PlutchikWheel(width=self.WHEEL_W, height=self.WHEEL_H)
        self.epistemic_tracker = LocalEpistemicTracker(window_frames=90, min_frames=20)
        self.mouse_tracker = MouseReadingAnalyzer()
        self.gaze_tracker = GazeReadingAnalyzer(window_size=5.0)
        
        # FUSED Dialog Controller
        self.dialog_controller = DialogController(dwell=dwell)
        
        # State tracking
        self.inference_mode = False
        self.current_gaze_score = 0.0
        self.calib_index = 0
        self.sampling_active = False
        
        # Drift Tracking
        self.drift_mode = False
        self.drift_index = 0

        # Unfiltered Raw Coordinate tracking
        self.gaze_x, self.gaze_y = self.SCREEN_W // 2, self.SCREEN_H // 2
        self.is_blinking = False
        self.blink_history = deque(maxlen=900)
        self.system_action = None

        # MediaPipe Headless Context
        self.landmark_lock = threading.Lock()
        self.latest_landmarks, self.latest_blendshapes = None, None
        self.new_data_available = False
        
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='face_landmarker.task'), 
            num_faces=1, min_face_detection_confidence=0.5, min_tracking_confidence=0.5, 
            output_face_blendshapes=True, running_mode=vision.RunningMode.LIVE_STREAM, 
            result_callback=self._mp_callback)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        self.mouse_tracker.start()
        self.last_frame_id = -1
        self.display_image = None
        self.running = True

    def _mp_callback(self, result, output_image, timestamp_ms):
        with self.landmark_lock:
            if result.face_blendshapes:
                self.latest_blendshapes = result.face_blendshapes[0]
                self.new_data_available = True

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

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            self.detector.detect_async(mp_image, int(time.time() * 1000))
            self.last_frame_id = current_frame_id

    def _update_and_fuse(self):
        if self.display_image is None: return

        process_new_frame = False
        with self.landmark_lock:
            if self.new_data_available:
                bs = self.latest_blendshapes
                self.new_data_available, process_new_frame = False, True

        mouse_score = self.mouse_tracker.get_score()
        emotion_state = self.emotion_detector.update(bs) if process_new_frame else self.emotion_detector.scores
        
        epistemic_state = {}
        if process_new_frame:
            self.epistemic_tracker.update(bs)
            epistemic_state = getattr(self.epistemic_tracker, 'current_state', {}) 
            
            # Blink & Fatigue tracking
            blink_l = next((cat.score for cat in bs if cat.category_name == 'eyeBlinkLeft'), 0.0)
            blink_r = next((cat.score for cat in bs if cat.category_name == 'eyeBlinkRight'), 0.0)
            if (blink_l + blink_r) / 2.0 > BLINK_BLENDSHAPE_THRESHOLD:
                self.is_blinking = True
                self.blink_history.append(1)
            else:
                self.is_blinking = False
                self.blink_history.append(0)

        if len(self.blink_history) > 100:
            perclos_score = sum(self.blink_history) / len(self.blink_history)
            if perclos_score > 0.15: 
                self.dialog_controller.system_message = "INTERVENTION: High Fatigue Detected. Consider a screen break."
                self.dialog_controller.help_active = True

        # Process RAW Gaze Data
        if self.gaze_reader.is_calibrated:
            raw_norm_x, raw_norm_y = self.gaze_reader.get_gaze_norm()
            self.gaze_x = int(raw_norm_x * self.SCREEN_W)
            self.gaze_y = int(raw_norm_y * self.SCREEN_H)
            
            self.heatmap.update(self.gaze_x, self.gaze_y)
            self.current_gaze_score = self.gaze_tracker.process_point(self.gaze_x, self.gaze_y)
        else:
            self.current_gaze_score = 0.0

        # Collision detection (only actively checked if in PDF reading mode)
        self.stable_hov = None
        if self.inference_mode and self.gaze_reader.is_calibrated:
            pg_img, paras = self.pdf.get_page(self.page)
            ph, pw = pg_img.shape[:2]
            ox = max(0, (self.SCREEN_W - pw) // 2)
            gpx = self.gaze_x - ox
            gpy = self.gaze_y + self.scroll_y
            hov = None
            
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

        # Fused Evaluation passing the hovered paragraph directly to the controller
        self.system_action = self.dialog_controller.evaluate(
            mouse_score, self.current_gaze_score, epistemic_state, emotion_state, self.stable_hov
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
                self._draw_fusion_hud(canvas, self.mouse_tracker.get_score(), self.system_action)
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

            # Draw HUD from Multimodal Dashboard
            if self.system_action and self.system_action['help_active']:
                cv2.putText(canvas, f"MULTIMODAL SYS: {self.system_action['message']}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

            if self.is_blinking:
                cv2.putText(canvas, "BLINK DETECTED", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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
        
        status_text = f"RAW Gaze Output: X({self.gaze_x}) Y({self.gaze_y})"
        if self.is_blinking: status_text += " [BLINKING]"
        cv2.putText(canvas, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255) if self.is_blinking else (200,200,200), 1)

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
        self.detector.close()
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