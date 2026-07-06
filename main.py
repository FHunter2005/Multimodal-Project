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
from pdf_context import get_help_text_for_paragraph
# External Modalities (Assumed existing)
from local_epistemic_tracker import LocalEpistemicTracker
from emotion_wheel import EmotionDetector, PlutchikWheel
from mouse_analyzer import MouseReadingAnalyzer
from gaze_analyzer import GazeReadingAnalyzer
from gaze_core import GazeReader, CALIB_PTS, DRIFT_PTS

# Newly Modularized Files
from ui_components import (SCREEN_W, SCREEN_H, SANDBOX_W, SANDBOX_H, WHEEL_W, WHEEL_H,
                           SAMPLES_NEEDED, BLINK_BLENDSHAPE_THRESHOLD, SCROLL_STEP,
                           draw_target_dot, GazeHeatmap)
from pdf_reader import PDFDocument, summarize, draw_dialog
from dialog_controller import DialogController
from face_modality import FaceModalityTracker
from voice_assistant import VoiceAssistant


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
        self.current_hover_idx = None
        self.hover_switch_time = 0.0
        self.recent_face_widths = deque(maxlen=10)
        self.recent_eye_openness = deque(maxlen=10)
        self.dlg_active = False
        self.dlg_para = None
        self.vertical_gaze_history = deque(maxlen=45) # ~1.5 seconds of blendshape frames
        self.last_scroll_time = 0.0
        self.dlg_summary = {'text': None}
        
        self.current_gaze_data = None
        self.gaze_history = deque(maxlen=10)
        self.off_screen_frames = 0
        self.gaze_wandering_frames = 0
        self.head_turned_away = False
        self.face_looking_at_screen = True
        self.face_direction_baseline = None
        self.face_direction_samples = deque(maxlen=30)
        self.calib_pose_samples = []
        self.baseline_yaw = 0.0
        self.baseline_pitch = 0.0
        self.baseline_roll = 0.0
        self.head_distraction_score = 0.0
        # Initialize Core Trackers
        self.gaze_reader = GazeReader(screen_w=SCREEN_W, screen_h=SCREEN_H, cam_src=0).start()
        self.face_tracker = FaceModalityTracker(blink_threshold=BLINK_BLENDSHAPE_THRESHOLD)
        self.prompt_active = False
        self.prompt_text = ""
        self.heatmap = GazeHeatmap()
        self.emotion_detector = EmotionDetector()
        self.emotion_wheel = PlutchikWheel(width=self.WHEEL_W, height=self.WHEEL_H)
        self.epistemic_tracker = LocalEpistemicTracker(window_frames=90, min_frames=20)
        self.mouse_tracker = MouseReadingAnalyzer()
        self.gaze_tracker = GazeReadingAnalyzer(window_size=5.0)
        self.mouse_score = None
        # FUSED Dialog Controller
        self.voice_assistant = VoiceAssistant(enabled=True)
        self.dialog_controller = DialogController(dwell=dwell, voice_assistant=VoiceAssistant(enabled=False))
        self.max_score = None
        # State tracking
        self.inference_mode = True
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
        self.face_tracker.start_calibration()
        self.baseline_finished = False
        self.mouse_tracker.start()
        self.last_frame_id = -1
        self.display_image = None
        self.running = True

    def run(self):
        print("Starting Unified Multimodal PDF Reader...")
        time.sleep(1.0)
        cv2.namedWindow('Reader & Dashboard', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Reader & Dashboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback('Reader & Dashboard', self._mouse_callback)
        while self.running:
            self._process_camera_feed()
            self._update_and_fuse()
            self._render()
            self._handle_input()
            
        self.cleanup()

    def _estimate_head_turn(self, lm):
        if lm is None:
            return False

        left_eye = np.array([(lm[159].x + lm[145].x) / 2.0, (lm[159].y + lm[145].y) / 2.0], dtype=np.float32)
        right_eye = np.array([(lm[386].x + lm[374].x) / 2.0, (lm[386].y + lm[374].y) / 2.0], dtype=np.float32)
        nose = np.array([lm[1].x, lm[1].y], dtype=np.float32)
        eye_mid = (left_eye + right_eye) / 2.0
        horizontal_offset = abs(nose[0] - eye_mid[0])
        vertical_offset = abs(nose[1] - eye_mid[1])
        return horizontal_offset > 0.08 and (abs(nose[0] - 0.5) > 0.05 or vertical_offset > 0.10)

    def _update_face_direction_baseline(self, lm):
        if lm is None:
            return

        nose = np.array([lm[1].x, lm[1].y], dtype=np.float32)
        self.face_direction_samples.append(nose)
        if len(self.face_direction_samples) < 20:
            return

        baseline = np.mean(list(self.face_direction_samples), axis=0)
        if self.face_direction_baseline is None:
            self.face_direction_baseline = baseline
            return

        self.face_direction_baseline = 0.8 * self.face_direction_baseline + 0.2 * baseline

    def _is_face_looking_at_screen(self, lm):
        if lm is None:
            return True

        self._update_face_direction_baseline(lm)
        if self.face_direction_baseline is None:
            return True

        nose = np.array([lm[1].x, lm[1].y], dtype=np.float32)
        dx = nose[0] - self.face_direction_baseline[0]
        dy = nose[1] - self.face_direction_baseline[1]
        angle_threshold = 0.12
        return abs(dx) <= angle_threshold and abs(dy) <= angle_threshold

    def _estimate_gaze_context(self):
        off_screen = (
            self.gaze_x < -80 or self.gaze_x > self.SCREEN_W + 80 or
            self.gaze_y < -80 or self.gaze_y > self.SCREEN_H + 80
        )

        self.gaze_history.append((self.gaze_x, self.gaze_y))
        gaze_wandering = False
        if len(self.gaze_history) >= 3:
            points = list(self.gaze_history)
            total_motion = 0.0
            direction_changes = 0
            prev_dx = prev_dy = None
            for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
                dx = x2 - x1
                dy = y2 - y1
                total_motion += np.sqrt(dx * dx + dy * dy)
                if prev_dx is not None:
                    if (dx > 0) != (prev_dx > 0) or (dy > 0) != (prev_dy > 0):
                        direction_changes += 1
                prev_dx, prev_dy = dx, dy
            gaze_wandering = (total_motion > 220) and (direction_changes >= 2)

        if off_screen:
            self.off_screen_frames += 1
        else:
            self.off_screen_frames = max(0, self.off_screen_frames - 1)

        if gaze_wandering:
            self.gaze_wandering_frames += 1
        else:
            self.gaze_wandering_frames = max(0, self.gaze_wandering_frames - 1)

        return {
            "gaze_off_screen": self.off_screen_frames >= 4,
            "gaze_wandering": self.gaze_wandering_frames >= 4,
        }

    def _estimate_mouse_wandering(self):
        mouse_history = list(getattr(self.mouse_tracker, 'history', []))
        if len(mouse_history) < 4:
            return False

        points = [(x, y) for _, x, y in mouse_history[-8:]]
        total_motion = 0.0
        direction_changes = 0
        prev_dx = prev_dy = None
        for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
            dx = x2 - x1
            dy = y2 - y1
            total_motion += np.sqrt(dx * dx + dy * dy)
            if prev_dx is not None:
                if (dx > 0) != (prev_dx > 0) or (dy > 0) != (prev_dy > 0):
                    direction_changes += 1
            prev_dx, prev_dy = dx, dy

        return total_motion > 600 and direction_changes >= 4

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

        # 1. Ask FaceModalityTracker for the latest data (now including the matrix)
        process_new_frame, bs, lm, matrix = self.face_tracker.update_state()
        if process_new_frame and bs:
            # 1. Extract vertical eye look directions
            bs_dict = {b.category_name: b.score for b in bs}
            down = (bs_dict.get('eyeLookDownLeft', 0.0) + bs_dict.get('eyeLookDownRight', 0.0)) / 2.0
            up = (bs_dict.get('eyeLookUpLeft', 0.0) + bs_dict.get('eyeLookUpRight', 0.0)) / 2.0
            
            # 2. Append the vertical gaze proxy
            self.vertical_gaze_history.append(down - up)
        # --- RESTORED VARIABLES ---
        # Process Modalities (update emotions/mouse regardless of matrix)
        self.mouse_score = self.mouse_tracker.get_data()
        emotion_state = self.emotion_detector.update(bs) if (process_new_frame and bs) else self.emotion_detector.scores
        # --------------------------
        if self.gaze_reader.is_calibrated and not getattr(self, 'baseline_finished', False):
            self.face_tracker.stop_calibration()
            self.baseline_finished = True
        if process_new_frame:
            # 2. Extract true 3D head rotation
            yaw, pitch, roll = 0.0, 0.0, 0.0
            if matrix is not None:
                # Extract the top-left 3x3 rotation matrix from the 4x4 transform matrix
                rmat = matrix[:3, :3] 
                euler_angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                pitch, yaw, roll = euler_angles
            if not self.gaze_reader.is_calibrated and self.sampling_active:
                self.calib_pose_samples.append((yaw, pitch, roll))
            # Compute the baseline exact center immediately after calibration ends
            elif self.gaze_reader.is_calibrated and self.calib_pose_samples:
                self.baseline_yaw = sum(p[0] for p in self.calib_pose_samples) / len(self.calib_pose_samples)
                self.baseline_pitch = sum(p[1] for p in self.calib_pose_samples) / len(self.calib_pose_samples)
                self.baseline_roll = sum(p[2] for p in self.calib_pose_samples) / len(self.calib_pose_samples)
                self.calib_pose_samples.clear() # Clear so this only runs once

            # Compute deviation from the user's personalized baseline
            if self.gaze_reader.is_calibrated:
                delta_yaw = abs(yaw - self.baseline_yaw)
                delta_pitch = abs(pitch - self.baseline_pitch)
                delta_roll = abs(roll - self.baseline_roll)
                
                # Flag if they look entirely away
                self.head_turned_away = delta_yaw > 15.0 or delta_pitch > 18.0 or delta_roll > 15.0
                self.face_looking_at_screen = not self.head_turned_away
                
                # Create a 0.0 -> 1.0 distraction score. 
                # (Allows 10 degrees of natural wiggle room before penalizing)
                max_dev = max(delta_yaw, delta_pitch, delta_roll)
                self.head_distraction_score = min(1.0, max(0.0, (max_dev - 10.0) / 10.0))
            else:
                self.head_turned_away = abs(yaw) > 15.0 or abs(pitch) > 20.0
                self.face_looking_at_screen = not self.head_turned_away
            # -------------------------------------

            # 4. Feed the pose data to your Epistemic Tracker!
            face_scale = getattr(self, 'debug_face_width', 0.5)
            if bs: 
                self.epistemic_tracker.update(bs, head_pose=(yaw, pitch, roll, face_scale))
            # 3. Replace your old 2D coordinate distance math with true 3D angles
          

        # Fetch the state unconditionally so it doesn't wipe on intermediate frames
        epistemic_state = getattr(self.epistemic_tracker, 'current_state', {}) 
            
        # --- MODIFIED: Fatigue tracking and new Voice Assistant trigger ---
  
        # --- MODIFIED: Fatigue tracking and new Voice Assistant trigger ---
        if self.face_tracker.is_fatigued(): 
            self.dialog_controller.system_message = "INTERVENTION: High Fatigue Detected. Consider a screen break."
            self.dialog_controller.help_active = True
            
            current_time = time.time()
            
            if not getattr(self, 'prompt_active', False) and (current_time - getattr(self, 'last_fatigue_spoken_time', 0.0)) > 60.0:
                self.voice_assistant.speak("It seems like you're quite tired. Let's take a quick break and in the meantime I will create a summary of what you read?")
                
                self.prompt_active = True
                self.prompt_text = "Take break & summarize? (Y/N)"
                
                self.last_fatigue_spoken_time = current_time
                self.last_spoken_time = current_time
        # ----------------------------------------------------------------
        # ----------------------------------------------------------------
        # ----------------------------------------------------------------
            
        # ... (Keep your existing RAW Gaze Data and Collision Detection code here) ...
        # 3. Process RAW Gaze Data
        self.stable_hov = None
        if self.gaze_reader.is_calibrated:
            raw_norm_x, raw_norm_y = self.gaze_reader.get_gaze_norm()
            self.gaze_x = int(raw_norm_x * self.SCREEN_W)
            self.gaze_y = int(raw_norm_y * self.SCREEN_H)

            self.heatmap.update(self.gaze_x, self.gaze_y)

            pg_img, paras = self.pdf.get_page(self.page)
            ph, pw = pg_img.shape[:2]
            ox = max(0, (self.SCREEN_W - pw) // 2)
            gpx = self.gaze_x - ox
            gpy = self.gaze_y + self.scroll_y
            on_paper = (0 <= gpx <= pw) and (0 <= gpy <= ph)

            self.current_gaze_data = self.gaze_tracker.process_point(self.gaze_x, self.gaze_y, on_paper)
            # 1. Base State Mapping
            self.current_gaze_data.update({
                "x": self.gaze_x,
                "y": self.gaze_y,
            })
            self.current_gaze_score =  self.current_gaze_data['score']
            self.current_gaze_state = self.current_gaze_data['state'] 

            # 2. Skimming Override
            if len(self.vertical_gaze_history) >= 15:
                gaze_variance = float(np.std(self.vertical_gaze_history))
                eye_skimming_score = min(1.0, gaze_variance / 0.06) 
                time_since_scroll = time.time() - getattr(self, 'last_scroll_time', 0.0)
                is_scrolling = time_since_scroll < 1.0
                
                if is_scrolling:
                    final_skimming_score = min(1.0, eye_skimming_score * 1.5 + 0.4)
                else:
                    final_skimming_score = eye_skimming_score
                    
                if final_skimming_score > 0.65:
                    self.current_gaze_state = "Skimming"

            # 3. Head Posture Distraction Override
            # 3. Head Posture Distraction Override (BUFFERED)
            # Increment a counter if they look away, decay it quickly if they look back
            if getattr(self, 'head_distraction_score', 0.0) > 0.60:
                self.head_away_frames = getattr(self, 'head_away_frames', 0) + 1
            else:
                self.head_away_frames = max(0, getattr(self, 'head_away_frames', 0) - 2)

            # [TIME USER: LOOKING AWAY]
            if getattr(self, 'head_away_frames', 0) > 200:
                self.current_gaze_state = "Distracted"
                self.current_gaze_score = max(self.current_gaze_score, self.head_distraction_score)

            # 4. EXPLICIT VOICE ASSISTANT LOGIC (Moved to evaluate finalized state)
            current_time = time.time()
            finalized_state = self.current_gaze_state

            # Calculate strict cooldown (default to 45 seconds)
            time_since_last_spoken = current_time - getattr(self, 'last_spoken_time', 0.0)

            if finalized_state in ["Distracted", "Thinking (Off-Text)"]:
                # ONLY trigger if no prompt is currently active AND cooldown has passed
                if not getattr(self, 'prompt_active', False) and time_since_last_spoken > 45.0:
                    
                    if finalized_state == "Distracted":
                        self.voice_assistant.speak("Would you like me to summarize the paper for you?")
                        self.prompt_text = "Summarize the paper? (Y/N)"
                    elif finalized_state == "Thinking (Off-Text)":
                        self.voice_assistant.speak("Would you like me to explain to you the current part of the paper?")
                        self.prompt_text = "Explain current part? (Y/N)"
                    
                    self.prompt_active = True
                    self.last_spoken_state = finalized_state
                    
                    # Sync global cooldowns
                    self.last_spoken_time = current_time
                    self.last_fatigue_spoken_time = current_time 

            elif finalized_state in ["Reading (Focused)", "Deep Focus", "Skimming", "Scanning (Upwards)"]:
                self.last_spoken_state = None

        
            # -------------------------------------------
            # --- NEW BLENDSHAPE SKIMMING LOGIC ---
            if len(self.vertical_gaze_history) >= 15:
                # 1. Standard deviation measures vertical "up and down" movement
                gaze_variance = float(np.std(self.vertical_gaze_history))
                
                # 2. Normalize variance to a 0-1 scale. (0.06 variance is very high movement)
                eye_skimming_score = min(1.0, gaze_variance / 0.06) 
                
                # 3. Increase significantly if user is scrolling
                time_since_scroll = time.time() - getattr(self, 'last_scroll_time', 0.0)
                is_scrolling = time_since_scroll < 1.0 # Scrolled in the last 1 second
                
                if is_scrolling:
                    # Boost the score and artificially inflate the base if scrolling
                    final_skimming_score = min(1.0, eye_skimming_score * 1.5 + 0.4)
                else:
                    final_skimming_score = eye_skimming_score
                    
                # 4. Override coordinate-based state if skimming is detected
                if final_skimming_score > 0.65:
                    self.current_gaze_state = "Skimming"
            if getattr(self, 'head_distraction_score', 0.0) > 0.60:
                self.current_gaze_state = "Distracted"
                # Boost the struggle score to match the physical deviation
                self.current_gaze_score = max(self.current_gaze_score, self.head_distraction_score)
            # 4. Collision detection (Hover state)
            if self.inference_mode:
                gaze_context = self._estimate_gaze_context()
                invalid_for_paragraph = (
                    not self.dlg_active and paras and pw > 0 and (
                        not on_paper or self.head_turned_away or gaze_context.get('gaze_off_screen', False)
                    )
                )

                # Replace your existing 'invalid_for_paragraph' block and subsequent loop with this:

            if invalid_for_paragraph:
                self.stable_hov = None
                self.hover_switch_time = time.time()
                if self.dialog_controller.cur_para is not None:
                    self.dialog_controller.reset_dwell()
            elif not self.dlg_active and paras and pw > 0:
                
                keep_current = False
                current_hover = None
                
                # 1. Hysteresis Check: If we already have a hovered paragraph, check if we are still in its "sticky" zone
                if self.stable_hov is not None and self.stable_hov < len(paras):
                    bx0, by0, bx1, by1 = paras[self.stable_hov]["bbox"]
                    
                    # Apply a generous 'sticky' margin to prevent flickering
                    sticky_margin_x = max(120, (bx1 - bx0) * 0.3) 
                    sticky_margin_y = 60 
                    
                    if (bx0 - sticky_margin_x <= gpx <= bx1 + sticky_margin_x) and \
                    (by0 - sticky_margin_y <= gpy <= by1 + sticky_margin_y):
                        current_hover = self.stable_hov
                        keep_current = True

                # 2. Only search for a new paragraph if we broke out of the sticky zone
                if not keep_current:
                    best_idx = None
                    best_dist = float('inf')
                    for pi, para in enumerate(paras):
                        bx0, by0, bx1, by1 = para["bbox"]
                        x_margin = max(80, (bx1 - bx0) * 0.2)
                        y_center = (by0 + by1) / 2.0
                        
                        # Standard, tighter bounding box for initial selection
                        if bx0 - x_margin <= gpx <= bx1 + x_margin:
                            vd = abs(gpy - y_center)
                            if vd < best_dist:
                                best_dist = vd
                                best_idx = pi

                    current_hover = best_idx if best_idx is not None and best_dist < 140 else None

                # 3. Time-gated switching logic (Keep your existing debounce logic)
                now = time.time()
                if current_hover is None:
                    self.stable_hov = None
                    self.hover_switch_time = now
                elif self.stable_hov is None:
                    self.stable_hov = current_hover
                    self.hover_switch_time = now
                elif current_hover != self.stable_hov and (now - self.hover_switch_time) > 0.15:
                    self.stable_hov = current_hover
                    self.hover_switch_time = now

        if self.gaze_reader.is_calibrated:
    
            gaze_context = self._estimate_gaze_context()
            mouse_wandering = self._estimate_mouse_wandering()

            # 5. DIALOG CONTROLLER (Fused State Evaluation)
            physical_cues = {
                'leaning_in': getattr(self, 'is_leaning_in', False),
                'squinting': getattr(self, 'is_squinting', False),
                'head_turned_away': self.head_turned_away,
                'face_looking_at_screen': self.face_looking_at_screen,
                'gaze_off_screen': gaze_context.get('gaze_off_screen', False),
                'gaze_wandering': gaze_context.get('gaze_wandering', False),
                'mouse_wandering': mouse_wandering,
            }
            self.system_action = self.dialog_controller.evaluate(
                self.mouse_score, self.current_gaze_data, epistemic_state, emotion_state, self.stable_hov, physical_cues
            )



        # 7. Generate PDF Summaries if needed
        # Safely check system_action (it might be None during calibration)
        # 7. Generate PDF Summaries if needed
        # 7. Generate PDF Summaries if needed
        triggered_para_idx = self.system_action.get("triggered_para") if self.system_action else None
        needs_summary = self.system_action.get("needs_summary", False) if self.system_action else False
        
        if triggered_para_idx is not None and needs_summary and not self.dlg_active and self.inference_mode:
            # Check if a prompt isn't already on the screen
            if not getattr(self, 'prompt_active', False):
                # 1. Ask the user via voice and visual prompt
                self.voice_assistant.speak("It seems like you might be stuck. Would you like a summary of this paragraph?")
                self.prompt_text = "Show paragraph summary? (Y/N)"
                self.prompt_active = True
                
                # 2. Save the exact paragraph index they were struggling on
                self.pending_help_para_idx = triggered_para_idx
                
               # def _summ(text, box): box['text'] = summarize(text)
               # threading.Thread(target=_summ, args=(self.dlg_para["text"], self.dlg_summary), daemon=True).start()
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
            # --- NEW: MID-LEFT DEBUG WINDOW ---
            debug_w = 360
            debug_h = 280
            dx = 20
            dy = (self.SCREEN_H - debug_h) // 2  # Vertically centered on the left
            
            # Create a semi-transparent overlay
            overlay = canvas.copy()
            cv2.rectangle(overlay, (dx, dy), (dx + debug_w, dy + debug_h), (20, 20, 25), -1)
            cv2.rectangle(overlay, (dx, dy), (dx + debug_w, dy + debug_h), (0, 255, 100), 2)
            cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas) # 85% opacity
            
            cv2.putText(canvas, "DEBUG: TUNNEL STATE", (dx + 15, dy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
            cv2.line(canvas, (dx, dy + 40), (dx + debug_w, dy + 40), (0, 255, 100), 1)
            
            # Tunnel-specific debug values only
            tunnel_active = getattr(self, 'focus_tunnel_active', False)
            tunnel_strength = getattr(self, 'focus_strength', 0.0)
            fused_state = self.system_action.get('fused_state', '') if self.system_action else ''
            fused_intensity = self.system_action.get('fused_intensity', 0.0) if self.system_action else 0.0

            lines = [
                f"Tunnel Active: {tunnel_active}",
                f"Strength: {tunnel_strength:.2f}",
                f"Fused State: {fused_state}",
                f"Fused Intensity: {fused_intensity:.2f}",
            ]

            # Top-right state intensity panel
            if self.system_action:
                state_intensities = self.dialog_controller.smoothed_intensities
                panel_x = self.SCREEN_W - 360
                panel_y = 20
                panel_w = 330
                panel_h = 170
                cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 25), -1)
                cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 255, 100), 1)
                cv2.putText(canvas, "STATE INTENSITIES", (panel_x + 12, panel_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)
                for i, (state, val) in enumerate(state_intensities.items()):
                    y = panel_y + 46 + i * 18
                    if y < panel_y + panel_h - 12:
                        cv2.putText(canvas, f"{state}: {val:.2f}", (panel_x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
            # Draw text with color coding for booleans
            for i, line in enumerate(lines):
                color = (200, 200, 200) # Default gray
                if "True" in line: 
                    color = (50, 255, 50) # Green for True
                elif "False" in line: 
                    color = (50, 50, 255) # Red for False
                
                cv2.putText(canvas, line, (dx + 15, dy + 70 + (i * 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
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
            (tw, th), _ = cv2.getTextSize(tracker_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
            cv2.rectangle(canvas, (10, 10), (20 + tw + 10, 20 + th + 15), (30, 30, 30), -1)
            cv2.rectangle(canvas, (10, 10), (20 + tw + 10, 20 + th + 15), (0, 255, 255), 1)
            cv2.putText(canvas, tracker_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Draw HUD from Multimodal Dashboard (Shifted down to avoid overlapping the tracker)
            if self.system_action and self.system_action['help_active']:
                cv2.putText(canvas, f"MULTIMODAL SYS: {self.system_action['message']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            if self.system_action and self.system_action.get('profile', 'Profile is being calibrated'):
                cv2.putText(canvas, f"USER PROFILE: {self.system_action.get('profile', 'Profile is being calibrated')}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            if self.face_tracker.is_blinking:
                cv2.putText(canvas, "BLINK DETECTED", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if self.dlg_active:
                draw_dialog(canvas, self.dlg_summary)
            if getattr(self, 'prompt_active', False) and not self.dlg_active:
                pw, ph = 360, 110
                px = self.SCREEN_W - pw - 30
                py = self.SCREEN_H - ph - 30
                
                # Create a semi-transparent dark background
                overlay = canvas.copy()
                cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (25, 25, 35), -1)
                cv2.addWeighted(overlay, 0.95, canvas, 0.05, 0, canvas)
                
                # Draw borders and header
                cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (0, 200, 255), 2)
                cv2.putText(canvas, "SYSTEM ASSISTANT", (px + 15, py + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                cv2.line(canvas, (px, py + 35), (px + pw, py + 35), (0, 200, 255), 1)
                
                # Draw the specific prompt and instructions
                cv2.putText(canvas, self.prompt_text, (px + 15, py + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
                cv2.putText(canvas, "[Y] Yes    [N] No", (px + 15, py + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 100), 2)
            # ---------------------------------------

        cv2.imshow('Reader & Dashboard', canvas)
    
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL and self.inference_mode and not self.dlg_active:
            self.last_scroll_time = time.time() 
            
            if flags > 0: # Scrolling UP
                if self.scroll_y == 0 and self.page > 0:
                    # Turn to previous page and jump to the bottom
                    self.page -= 1
                    self.scroll_y = 99999 # _render() will automatically clamp this to max_scroll
                    self.dialog_controller.reset_dwell()
                    self.hov_buf.clear()
                else:
                    self.scroll_y = max(0, self.scroll_y - SCROLL_STEP)
                    
            else: # Scrolling DOWN
                if self.scroll_y >= self.max_scroll and self.page < self.pdf.n - 1:
                    # Turn to next page and jump to the top
                    self.page += 1
                    self.scroll_y = 0
                    self.dialog_controller.reset_dwell()
                    self.hov_buf.clear()
                else:
                    self.scroll_y = min(self.max_scroll, self.scroll_y + SCROLL_STEP)

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
        
        # Check if mouse_score is a valid dictionary and extract the intensity
        if isinstance(mouse_score, dict):
            m_intensity = mouse_score.get("intensity", 0.0)
            mouse_text = f"Mouse PAR Score: {m_intensity:.2f} [{mouse_score.get('state', 'Unknown').lower()}]"
            mouse_color = (200, 200, 200)
        else:
            mouse_text = "Mouse PAR Score: INACTIVE"
            mouse_color = (100, 100, 100)
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
        # --- NEW: Prompt Inputs (Bottom Right Window) ---
        # --- NEW: Prompt Inputs (Bottom Right Window) ---
        # --- NEW: Prompt Inputs (Bottom Right Window) ---
        elif key_char in (ord('y'), ord('Y')) and getattr(self, 'prompt_active', False):
            self.prompt_active = False
            
            _, paras = self.pdf.get_page(self.page)
            if paras:
                self.dlg_active = True
                
                # 1. Check if we saved a specific struggling paragraph
                if getattr(self, 'pending_help_para_idx', None) is not None:
                    target_idx = self.pending_help_para_idx
                    self.pending_help_para_idx = None # Clear it after using
                else:
                    # Fallback to the hovered paragraph if it was a different prompt (like fatigue)
                    target_idx = self.stable_hov if self.stable_hov is not None else 0
                    
                self.dlg_para = paras[target_idx] if target_idx < len(paras) else paras[0]
                
                # --- ATOMIC CONTEXT LOOKUP ---
                self.dlg_summary['text'] = get_help_text_for_paragraph(self.page, target_idx)
                # Spawn the Gemini thread
               # def _summ(text, box): box['text'] = summarize(text)
               # threading.Thread(target=_summ, args=(self.dlg_para["text"], self.dlg_summary), daemon=True).start()
                
        elif key_char in (ord('n'), ord('N')) and getattr(self, 'prompt_active', False):
            self.prompt_active = False
        # Reading Inputs
        elif not self.dlg_active:
            if key_char == ord('j'):
                self.scroll_y = min(self.max_scroll, self.scroll_y + SCROLL_STEP)
                self.last_scroll_time = time.time() # <-- ADD THIS
            elif key_char == ord('k'):
                self.scroll_y = max(0, self.scroll_y - SCROLL_STEP)
                self.last_scroll_time = time.time() # <-- ADD THIS
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
        cv2.setMouseCallback('Reader & Dashboard', lambda *args: None)
        self.gaze_reader.stop()
        self.face_tracker.stop()
        self.mouse_tracker.stop()
        self.pdf.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Multimodal PDF Reader Dashboard")
    #ap.add_argument('--pdf', default=None, help="Path to the PDF document to read.")
    ap.add_argument('--dwell', type=float, default=5.0, help="Dwell time in seconds to trigger stuck dialog.")
    ap.add_argument('--zoom', type=float, default=None, help="PDF render zoom multiplier.")
    
    args = ap.parse_args()
    args.pdf = 'example.pdf'
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
