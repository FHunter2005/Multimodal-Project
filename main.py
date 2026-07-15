
import cv2
import time
import numpy as np
import threading

import argparse
import sys
from collections import deque
from pathlib import Path
from pdf_context import get_help_text_for_paragraph

from local_epistemic_tracker import LocalEpistemicTracker
from emotion_wheel import EmotionDetector, PlutchikWheel
from mouse_analyzer import MouseReadingAnalyzer
from gaze_analyzer import GazeReadingAnalyzer
from gaze_core import GazeReader, CALIB_PTS, DRIFT_PTS
from voice_listener import VoiceListener

from ui_components import (SCREEN_W, SCREEN_H, SANDBOX_W, SANDBOX_H, WHEEL_W, WHEEL_H,
                           SAMPLES_NEEDED, BLINK_BLENDSHAPE_THRESHOLD, SCROLL_STEP,
                           draw_target_dot, GazeHeatmap)
from pdf_reader import PDFDocument, summarize, draw_dialog
from dialog_controller import DialogController
from face_modality import FaceModalityTracker
from voice_assistant import VoiceAssistant
from pdf_context import get_help_text_for_paragraph
from tutorial_manager import TutorialManager

class ReaderHelperApp:
    def __init__(self, pdf_path, zoom, dwell):
        self.SCREEN_W, self.SCREEN_H = SCREEN_W, SCREEN_H
        self.SANDBOX_W, self.SANDBOX_H = SANDBOX_W, SANDBOX_H
        self.WHEEL_W, self.WHEEL_H = WHEEL_W, WHEEL_H
        
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
        self.vertical_gaze_history = deque(maxlen=45)
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
     
        self.tutorial = TutorialManager(self.SCREEN_W, self.SCREEN_H)
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
 
        self.voice_assistant = VoiceAssistant(enabled=True)
      
        self.voice_listener = VoiceListener()
        

        self.voice_listener.start_continuous(self._on_voice_heard)
        self.pending_voice_action = None
        self.dialog_controller = DialogController(dwell=dwell, voice_assistant=VoiceAssistant(enabled=False))
        self.max_score = None
   
        self.inference_mode = True
        self.current_gaze_score = 0.0
        self.current_gaze_state = "Initializing"
        self.calib_index = 0
        self.sampling_active = False
        self.last_scroll_time = 0.0
        self.scroll_active_duration = 0.0
        self.last_fuse_time = time.time()
        # Drift Tracking
        self.drift_mode = False
        self.drift_index = 0

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
    def _on_voice_heard(self, text):
        """Callback from the background STT thread."""
        
        if getattr(self, 'tutorial', None) and self.tutorial.is_active:
            self.pending_voice_action = f"TUTORIAL:{text}"
            return

        if getattr(self, 'prompt_active', False):
            if any(w in text for w in ["yes", "yeah", "sure", "yep", "ok", "please"]):
                self.pending_voice_action = 'ACCEPT_PROMPT'
            elif any(w in text for w in ["no", "nope", "nah", "stop", "cancel"]):
                self.pending_voice_action = 'DECLINE_PROMPT'
            return

       
        if ("summarize" in text or "summarise" in text) and ("this" in text or "paragraph" in text or "it" in text):
            # INSTEAD OF SUMMARIZE_HOVER, ask for confirmation
            self.pending_voice_action = 'ASK_CONFIRM_SUMMARY'
        elif "explain" in text and "this" in text:
            self.pending_voice_action = 'ASK_CONFIRM_SUMMARY'

    def _trigger_deictic_summary(self):

        _, paras = self.pdf.get_page(self.page)
        target_idx = None

     
        circled_center = self.mouse_tracker.detect_circling_gesture(time_window=2.0)
        
        if circled_center:
            cx, cy = circled_center
            
           
            pg_img, _ = self.pdf.get_page(self.page)
            ph, pw = pg_img.shape[:2]
            ox = max(0, (self.SCREEN_W - pw) // 2)
            
            gpx = cx - ox
            gpy = cy + self.scroll_y 
    
            for pi, para in enumerate(paras):
                bx0, by0, bx1, by1 = para["bbox"]
               
                margin = 80 
                if (bx0 - margin <= gpx <= bx1 + margin) and (by0 - margin <= gpy <= by1 + margin):
                    target_idx = pi
                    print(f"[SYSTEM] Mouse circle gesture detected at paragraph {pi}")
                    break

        if target_idx is None and self.stable_hov is not None:
            if self.stable_hov < len(paras):
                target_idx = self.stable_hov
                print(f"[SYSTEM] Gaze fixation utilized at paragraph {target_idx}")

    
        if target_idx is not None:
            self.dlg_para = paras[target_idx]
            self.dlg_active = True
            
            
            self.dlg_summary['text'] = get_help_text_for_paragraph(self.page, target_idx)
            
           
            self.voice_assistant.speak("Here is the context for the paragraph you selected.")
        else:
          
            self.voice_assistant.speak("I'm not sure which paragraph you mean. Please look at or circle the text and ask again.")
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


        cutoff = time.time() - 1.5
        points = [(x, y) for t, x, y in mouse_history if t >= cutoff]
        if len(points) < 4:
            return False

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
        now = time.time()
        dt = now - getattr(self, 'last_fuse_time', now)
        self.last_fuse_time = now
        

        if now - getattr(self, 'last_scroll_time', 0.0) < 0.25:
            self.scroll_active_duration = getattr(self, 'scroll_active_duration', 0.0) + dt
        else:
            self.scroll_active_duration = 0.0
       
        process_new_frame, bs, lm, matrix = self.face_tracker.update_state()
        if process_new_frame and bs:

            bs_dict = {b.category_name: b.score for b in bs}
            down = (bs_dict.get('eyeLookDownLeft', 0.0) + bs_dict.get('eyeLookDownRight', 0.0)) / 2.0
            up = (bs_dict.get('eyeLookUpLeft', 0.0) + bs_dict.get('eyeLookUpRight', 0.0)) / 2.0
            
          
            self.vertical_gaze_history.append(down - up)
     
        self.mouse_score = self.mouse_tracker.get_data()
        emotion_state = self.emotion_detector.update(bs) if (process_new_frame and bs) else self.emotion_detector.scores
       
        if self.gaze_reader.is_calibrated and not getattr(self, 'baseline_finished', False):
            self.face_tracker.stop_calibration()
            self.baseline_finished = True
        if process_new_frame:
 
            yaw, pitch, roll = 0.0, 0.0, 0.0
            if matrix is not None:
                rmat = matrix[:3, :3] 
                euler_angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                pitch, yaw, roll = euler_angles
            if not self.gaze_reader.is_calibrated and self.sampling_active:
                self.calib_pose_samples.append((yaw, pitch, roll))
      
            elif self.gaze_reader.is_calibrated and self.calib_pose_samples:
                self.baseline_yaw = sum(p[0] for p in self.calib_pose_samples) / len(self.calib_pose_samples)
                self.baseline_pitch = sum(p[1] for p in self.calib_pose_samples) / len(self.calib_pose_samples)
                self.baseline_roll = sum(p[2] for p in self.calib_pose_samples) / len(self.calib_pose_samples)
                self.calib_pose_samples.clear() 

        
            if self.gaze_reader.is_calibrated:
                delta_yaw = abs(yaw - self.baseline_yaw)
                delta_pitch = abs(pitch - self.baseline_pitch)
                delta_roll = abs(roll - self.baseline_roll)
                
               
                self.head_turned_away = delta_yaw > 15.0 or delta_pitch > 18.0 or delta_roll > 15.0
                self.face_looking_at_screen = not self.head_turned_away
                
 
                max_dev = max(delta_yaw, delta_pitch, delta_roll)
                self.head_distraction_score = min(1.0, max(0.0, (max_dev - 10.0) / 10.0))
            else:
                self.head_turned_away = abs(yaw) > 15.0 or abs(pitch) > 20.0
                self.face_looking_at_screen = not self.head_turned_away
           
            face_scale = getattr(self, 'debug_face_width', 0.5)
            if bs: 
                self.epistemic_tracker.update(bs, head_pose=(yaw, pitch, roll, face_scale))
      
        epistemic_state = getattr(self.epistemic_tracker, 'current_state', {}) 
            
   
        if self.face_tracker.is_fatigued(): 
            self.dialog_controller.system_message = "INTERVENTION: High Fatigue Detected. Consider a screen break."
            self.dialog_controller.help_active = True
            
            current_time = time.time()
            
            if not getattr(self, 'prompt_active', False) and (current_time - getattr(self, 'last_fatigue_spoken_time', 0.0)) > 60.0:
                self.voice_assistant.speak("It seems like you're quite tired. Let's take a quick break and in the meantime I will create a summary of what you read?")
                
                self.prompt_active = True
                
                self.prompt_text = "Take break & summarize? (Y/N)"
                self.voice_assistant.speak(
                    "It seems like you're quite tired. Let's take a quick break and in the meantime I will create a summary of what you read?",
                    
                )
                self.last_fatigue_spoken_time = current_time
                self.last_spoken_time = current_time
   

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
          
            self.current_gaze_data.update({
                "x": self.gaze_x,
                "y": self.gaze_y,
            })
            self.current_gaze_score =  self.current_gaze_data['score']
            self.current_gaze_state = self.current_gaze_data['state'] 

          
            if len(self.vertical_gaze_history) >= 15:
                recent_gaze = list(self.vertical_gaze_history)[-15:]
                instant_variance = float(np.std(recent_gaze))
                
              
                current_momentum = getattr(self, 'eye_skimming_momentum', 0.0)
                
                if instant_variance > 0.035: # Base threshold for "active" eye movement
                    growth_step = (instant_variance * 0.4) + (current_momentum * 0.08)
                    self.eye_skimming_momentum = min(1.0, current_momentum + growth_step)
                else:
                    self.eye_skimming_momentum = max(0.0, current_momentum - 0.15)
                    
                eye_skimming_score = self.eye_skimming_momentum
                
                time_since_scroll = time.time() - getattr(self, 'last_scroll_time', 0.0)
                is_scrolling_periodically = time_since_scroll < 4.0 # Scrolled within the last 4 seconds
                
                if is_scrolling_periodically:
                    final_skimming_score = min(1.0, eye_skimming_score * 1.5 + 0.3)
                else:
                    final_skimming_score = min(0.45, eye_skimming_score * 0.4)
                
                if getattr(self, 'scroll_active_duration', 0.0) > 1.0:
                    scroll_bypass_score = (self.scroll_active_duration - 1.0) * 1.5 
                    final_skimming_score = min(1.0, max(final_skimming_score, 0.5 + scroll_bypass_score))
                
                if final_skimming_score > 0.65:
                    self.current_gaze_state = "Skimming"

            if getattr(self, 'head_distraction_score', 0.0) > 0.60:
                self.head_away_frames = getattr(self, 'head_away_frames', 0) + 1
            else:
                self.head_away_frames = max(0, getattr(self, 'head_away_frames', 0) - 2)

            if getattr(self, 'head_away_frames', 0) > 200:
                self.current_gaze_state = "Thinking (Off-Text)" # Changed from "Distracted"
                self.current_gaze_score = max(self.current_gaze_score, self.head_distraction_score)

            current_time = time.time()
            finalized_state = self.current_gaze_state

            time_since_last_spoken = current_time - getattr(self, 'last_spoken_time', 0.0)

            if finalized_state in ["Distracted", "Thinking (Off-Text)"]:
                # ONLY trigger if no prompt is currently active AND cooldown has passed
                if not getattr(self, 'prompt_active', False) and time_since_last_spoken > 45.0:

                    if finalized_state == "Distracted":
                        self.voice_assistant.speak(
                            "You seem a bit distracted. Try to refocus on the page."
                        )
                    elif finalized_state == "Thinking (Off-Text)":
                        self.prompt_text = "Explain current part? (Y/N)"
                        self.voice_assistant.speak(
                            "Would you like me to explain to you the current part of the paper?",

                        )
                        self.prompt_active = True

                    self.last_spoken_state = finalized_state

                    # Sync global cooldowns
                    self.last_spoken_time = current_time
                    self.last_fatigue_spoken_time = current_time

            elif finalized_state in ["Reading (Focused)", "Deep Focus", "Skimming", "Scanning (Upwards)"]:
                self.last_spoken_state = None

        
          
            if getattr(self, 'head_distraction_score', 0.0) > 0.60:
                self.current_gaze_state = "Thinking (Off-Text)" # Changed from "Distracted"
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
            self.current_gaze_data['state'] = self.current_gaze_state
            self.current_gaze_data['score'] = self.current_gaze_score
            self.system_action = self.dialog_controller.evaluate(
                self.mouse_score, self.current_gaze_data, epistemic_state, emotion_state, self.stable_hov, physical_cues
            )
            # 4. EXPLICIT VOICE ASSISTANT LOGIC (Moved here to use the FUSED state)
            if self.system_action:
                fused_state = self.system_action.get('fused_state', 'Initializing')
                current_time = time.time()
                time_since_last_spoken = current_time - getattr(self, 'last_spoken_time', 0.0)

                if fused_state in ["Distracted", "Thinking (Off-Text)"]:
                    if not getattr(self, 'prompt_active', False) and time_since_last_spoken > 45.0:

                        if fused_state == "Distracted":
                            self.voice_assistant.speak("You seem a bit distracted. Try to refocus on the page.")
                        elif fused_state == "Thinking (Off-Text)":
                            self.prompt_text = "Explain current part? (Y/N)"
                            self.voice_assistant.speak("Would you like me to explain to you the current part of the paper?")
                            self.prompt_active = True

                        self.last_spoken_state = fused_state
                        self.last_spoken_time = current_time
                        self.last_fatigue_spoken_time = current_time


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
               # threading.Thread(target=_summ, args=(self.dlg_para["text"], self.dlg_summary), daemon=True).start()
    def _render(self):
        canvas = np.zeros((self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8)

        if not self.gaze_reader.is_calibrated or not self.inference_mode:
            if not self.gaze_reader.is_calibrated:
                canvas[:] = 18
                self._draw_calibration_ui(canvas, self.SCREEN_W, self.SCREEN_H)
            else:
                # 1. Base Framework: Sandbox, Emotion Wheel, Epistemic Tracker
                sandbox = np.ones((self.SANDBOX_H, self.SANDBOX_W, 3), dtype=np.uint8) * 18
                wheel_canvas = self.emotion_wheel.render(self.emotion_detector.scores)
                epistemic_canvas = np.zeros((self.WHEEL_H, self.WHEEL_W, 3), dtype=np.uint8)
                self.epistemic_tracker.render(epistemic_canvas)

                canvas[0:self.SANDBOX_H, 0:self.SANDBOX_W] = sandbox                     
                canvas[0:self.WHEEL_H, self.SANDBOX_W:self.SCREEN_W] = wheel_canvas               
                canvas[self.WHEEL_H:self.SCREEN_H, self.SANDBOX_W:self.SCREEN_W] = epistemic_canvas        
                
                self._draw_tracking_ui(canvas, self.SCREEN_W, self.SCREEN_H)
                self._draw_fusion_hud(canvas, self.mouse_score, self.system_action)
                cv2.putText(canvas, "Press [I] to return to PDF READER", (1320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                debug_w = 360
                debug_h = 140 
                dx = 20
                dy = (self.SCREEN_H - debug_h) // 2  # Vertically centered on the left
                
                overlay = canvas.copy()
                cv2.rectangle(overlay, (dx, dy), (dx + debug_w, dy + debug_h), (20, 20, 25), -1)
                cv2.rectangle(overlay, (dx, dy), (dx + debug_w, dy + debug_h), (0, 255, 100), 2)
                cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
                
                cv2.putText(canvas, "DEBUG: FUSED STATE", (dx + 15, dy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2, cv2.LINE_AA)
                cv2.line(canvas, (dx, dy + 40), (dx + debug_w, dy + 40), (0, 255, 100), 1)
                
                fused_state = self.system_action.get('fused_state', '') if self.system_action else ''
                fused_intensity = self.system_action.get('fused_intensity', 0.0) if self.system_action else 0.0

                lines = [
                    f"Fused State: {fused_state}",
                    f"Fused Intensity: {fused_intensity:.2f}",
                ]

                for i, line in enumerate(lines):
                    color = (200, 200, 200)
                    cv2.putText(canvas, line, (dx + 15, dy + 70 + (i * 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                if self.system_action:
                    state_intensities = self.dialog_controller.smoothed_intensities
                    panel_w = 330
                    panel_x = self.SANDBOX_W - panel_w - 20 
                    panel_y = 20
                    panel_h = 170
                    
                    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 25), -1)
                    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 255, 100), 1)
                    cv2.putText(canvas, "STATE INTENSITIES", (panel_x + 12, panel_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1, cv2.LINE_AA)
                    
                    for i, (state, val) in enumerate(state_intensities.items()):
                        y = panel_y + 46 + i * 20
                        if y < panel_y + panel_h - 12:
                            cv2.putText(canvas, f"{state}: {val:.2f}", (panel_x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

                # 4. MULTIMODAL SYS ALERTS
                sys_y = 90
                if self.system_action and self.system_action['help_active']:
                    cv2.putText(canvas, f"MULTIMODAL SYS: {self.system_action['message']}", (20, sys_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2, cv2.LINE_AA)
                if self.system_action and self.system_action.get('profile', None):
                    cv2.putText(canvas, f"USER PROFILE: {self.system_action.get('profile')}", (20, sys_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2, cv2.LINE_AA)
                if self.face_tracker.is_blinking:
                    cv2.putText(canvas, "BLINK DETECTED", (20, sys_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # -------------------------------------------------------------
        # CLEAN PDF READER MODE
        # -------------------------------------------------------------
        # -------------------------------------------------------------
        # CLEAN PDF READER MODE
        # -------------------------------------------------------------
        # -------------------------------------------------------------
        # CLEAN PDF READER MODE
        # -------------------------------------------------------------
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

            is_tracking = getattr(self.gaze_reader, 'face_detected', True)
            track_color = (0, 220, 50) if is_tracking else (50, 50, 220)
            track_text = "AI Assistant Active" if is_tracking else "Face Lost - Waiting..."
            pulse_r = int(5 + 2 * np.sin(time.time() * 4)) if is_tracking else 5
            
            cv2.circle(canvas, (self.SCREEN_W - 200, 35), pulse_r, track_color, -1)
            cv2.putText(canvas, track_text, (self.SCREEN_W - 185, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.putText(canvas, "Say 'summarize this' if you need a summary right away.", (60, self.SCREEN_H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 0, 0), 2, cv2.LINE_AA)
            btn_cx, btn_cy = self.SCREEN_W - 60, self.SCREEN_H - 60
            btn_r = 25
            
            # Draw Green Button
            cv2.circle(canvas, (btn_cx, btn_cy), btn_r, (0, 180, 80), -1)
            cv2.circle(canvas, (btn_cx, btn_cy), btn_r, (0, 255, 120), 2)
            
            # Draw "Status" Bar Chart Icon inside the button
            cv2.rectangle(canvas, (btn_cx - 10, btn_cy + 4), (btn_cx - 4, btn_cy + 10), (255,255,255), -1) # Short bar
            cv2.rectangle(canvas, (btn_cx - 3, btn_cy - 4), (btn_cx + 3, btn_cy + 10), (255,255,255), -1)  # Mid bar
            cv2.rectangle(canvas, (btn_cx + 4, btn_cy - 12), (btn_cx + 10, btn_cy + 10), (255,255,255), -1) # Tall bar

            if getattr(self, 'show_status_tab', False):
                tab_w, tab_h = 240, 80
                tab_x = btn_cx - tab_w + 20
                tab_y = btn_cy - btn_r - tab_h - 15
                
                # Dark Semi-Transparent Panel
                overlay = canvas.copy()
                cv2.rectangle(overlay, (tab_x, tab_y), (tab_x + tab_w, tab_y + tab_h), (30, 30, 35), -1)
                cv2.addWeighted(overlay, 0.95, canvas, 0.05, 0, canvas)
                cv2.rectangle(canvas, (tab_x, tab_y), (tab_x + tab_w, tab_y + tab_h), (0, 200, 100), 2)
                
                # Fetch Fused State safely
                fused_state = self.system_action.get('fused_state', 'Initializing...') if getattr(self, 'system_action', None) else 'Initializing...'
                
                cv2.putText(canvas, "YOUR CURRENT STATE:", (tab_x + 15, tab_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
                cv2.putText(canvas, fused_state, (tab_x + 15, tab_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2, cv2.LINE_AA)

            # Drift UI overlay
            if self.drift_mode:
                ddx = int(DRIFT_PTS[self.drift_index][0] * self.SCREEN_W)
                ddy = int(DRIFT_PTS[self.drift_index][1] * self.SCREEN_H)
                cv2.putText(canvas, f"Drift ({self.drift_index+1}/{len(DRIFT_PTS)}) - SPACE", (20, self.SCREEN_H-75), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 220, 50), 2, cv2.LINE_AA)
                
                if self.gaze_reader._is_drift and self.sampling_active:
                    prog = self.gaze_reader.calibration_progress
                    cv2.circle(canvas, (ddx, ddy), 15, (0, 255, 0), -1)
                    cv2.putText(canvas, f"Hold still... {int(prog * 100)}%", (ddx - 40, ddy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
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

            # Minimalist Page and Control UI
            scroll_pct = int(100 * self.scroll_y / self.max_scroll) if self.max_scroll > 0 else 0
            cv2.putText(canvas, f"Page {self.page+1}/{self.pdf.n}  [{scroll_pct}%]", (20, self.SCREEN_H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
            cv2.putText(canvas, "[I] Toggle Dashboard", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

            # Active Prompts & AI Overlays
            if self.dlg_active:
                draw_dialog(canvas, self.dlg_summary)
            if getattr(self, 'prompt_active', False) and not self.dlg_active:
                pw, ph = 360, 110
                px = self.SCREEN_W - pw - 30
                py = self.SCREEN_H - ph - 30
                
                overlay = canvas.copy()
                cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (25, 25, 35), -1)
                cv2.addWeighted(overlay, 0.95, canvas, 0.05, 0, canvas)
                
                cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (0, 200, 255), 2)
                cv2.putText(canvas, "SYSTEM ASSISTANT", (px + 15, py + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
                cv2.line(canvas, (px, py + 35), (px + pw, py + 35), (0, 200, 255), 1)
                cv2.putText(canvas, self.prompt_text, (px + 15, py + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, "[Y] Yes    [N] No", (px + 15, py + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 100), 2, cv2.LINE_AA)

        # Place this at the very bottom of def _render(self):
        if getattr(self, 'tutorial', None) and self.tutorial.is_active:
            canvas = self.tutorial.draw_overlay(canvas)
            
        cv2.imshow('Reader & Dashboard', canvas)

    def _mouse_callback(self, event, x, y, flags, param):
        # --- NEW: Intercept clicks for the Tutorial ---
        if getattr(self, 'tutorial', None) and self.tutorial.is_active:
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.tutorial.handle_click(x, y):
                    # If clicking "Next" finished the tutorial, start calibration immediately
                    if not self.tutorial.is_active:
                        self.face_tracker.start_calibration()
            return # Block all PDF/System mouse interactions while tutorial is active

        # 1. Handle Scrolling
        if event == cv2.EVENT_MOUSEWHEEL and self.inference_mode and not self.dlg_active:
            self.last_scroll_time = time.time() 
            
            if flags > 0: # Scrolling UP
                if self.scroll_y == 0 and self.page > 0:
                    self.page -= 1
                    self.scroll_y = 99999 
                    self.dialog_controller.reset_dwell()
                    self.hov_buf.clear()
                else:
                    self.scroll_y = max(0, self.scroll_y - SCROLL_STEP)
                    
            else: # Scrolling DOWN
                if self.scroll_y >= self.max_scroll and self.page < self.pdf.n - 1:
                    self.page += 1
                    self.scroll_y = 0
                    self.dialog_controller.reset_dwell()
                    self.hov_buf.clear()
                else:
                    self.scroll_y = min(self.max_scroll, self.scroll_y + SCROLL_STEP)

        # 2. Handle Clicks for the Status Button
        elif event == cv2.EVENT_LBUTTONDOWN and self.inference_mode:
            # Check if click falls within the radius of our status button
            btn_cx, btn_cy = self.SCREEN_W - 60, self.SCREEN_H - 60
            btn_r = 25
            
            # Simple circular collision math (x - cx)^2 + (y - cy)^2 <= r^2
            if (x - btn_cx)**2 + (y - btn_cy)**2 <= btn_r**2:
                # Toggle the status tab on/off
                self.show_status_tab = not getattr(self, 'show_status_tab', False)

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
            cv2.putText(canvas, text, ((w - text_w) // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2, cv2.LINE_AA)
        
        for i, (px, py) in enumerate(CALIB_PTS):
            cv2.circle(canvas, (int(px * w), int(py * h)), 8 if i < self.calib_index else 5 if i == self.calib_index else 6, (0, 200, 0) if i < self.calib_index else (255, 255, 255) if i == self.calib_index else (80, 80, 80), -1)

    def _draw_tracking_ui(self, canvas, w, h):
        self.heatmap.render(canvas)
        cv2.line(canvas, (self.gaze_x - 14, self.gaze_y), (self.gaze_x + 14, self.gaze_y), (255, 255, 255), 1)
        cv2.line(canvas, (self.gaze_x, self.gaze_y - 14), (self.gaze_x, self.gaze_y + 14), (255, 255, 255), 1)
        cv2.circle(canvas, (self.gaze_x, self.gaze_y), 4, (255, 255, 255), -1)
        
        cv2.putText(canvas, "Tracking Base — 'r' recalibrate  'q' quit", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        status_text = f"RAW Gaze Output: X({self.gaze_x}) Y({self.gaze_y}) | State: {self.current_gaze_state}"
        if self.face_tracker.is_blinking: status_text += " [BLINKING]"
        cv2.putText(canvas, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255) if self.face_tracker.is_blinking else (200,200,200), 1, cv2.LINE_AA)

    def _draw_fusion_hud(self, canvas, mouse_score, action_state):
        hud_x, hud_y = 30, self.SANDBOX_H - 120
        cv2.rectangle(canvas, (hud_x - 10, hud_y - 40), (hud_x + 550, hud_y + 100), (40, 40, 40), -1)
        
        color = (0, 100, 255) if action_state["help_active"] else (255, 255, 255)
        cv2.putText(canvas, f"SYSTEM: {action_state['message']}", (hud_x, hud_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        
        # Check if mouse_score is a valid dictionary and extract the intensity
        if isinstance(mouse_score, dict):
            m_intensity = mouse_score.get("intensity", 0.0)
            mouse_text = f"Mouse PAR Score: {m_intensity:.2f} [{mouse_score.get('state', 'Unknown').lower()}]"
            mouse_color = (200, 200, 200)
        else:
            mouse_text = "Mouse PAR Score: INACTIVE"
            mouse_color = (100, 100, 100)
        cv2.putText(canvas, mouse_text, (hud_x, hud_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouse_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Fusion Struggle Level: {action_state['struggle_level']:.2f}", (hud_x, hud_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    def _accept_prompt(self):
        self.prompt_active = False
        self.dialog_controller.reset_intervention()
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

    def _decline_prompt(self):
        self.prompt_active = False
        self.dialog_controller.reset_intervention()

    def _handle_input(self):
        key = cv2.waitKey(1)
        key_char = key & 0xFF
        
        if key_char == ord('q'): 
            self.running = False
            return

        if getattr(self, 'pending_voice_action', None) is not None:
            action = self.pending_voice_action
            self.pending_voice_action = None # Clear the flag immediately
            
            if action.startswith("TUTORIAL:"):
                text = action.split("TUTORIAL:")[1]
                if self.tutorial.is_active:
                    self.tutorial.process_input(None, voice_command=text)
                    if not self.tutorial.is_active:
                        self.face_tracker.start_calibration() 
            
            elif action == 'ASK_CONFIRM_SUMMARY':
                print("[SYSTEM] Asking for summary confirmation...")
                self.prompt_active = True
                self.prompt_reason = 'VOICE_COMMAND_SUMMARY' # Remember why we prompted
                self.prompt_text = "Summarize this paragraph? (Y/N)"
                self.voice_assistant.speak("Did you want me to summarize this paragraph?")
                
            elif action == 'ACCEPT_PROMPT' and getattr(self, 'prompt_active', False):
                print("[SYSTEM] Voice command accepted prompt.")
                if getattr(self, 'prompt_reason', None) == 'VOICE_COMMAND_SUMMARY':
                    self.prompt_active = False
                    self.prompt_reason = None
                    self._trigger_deictic_summary() # Execute the actual summary
                else:
                    self._accept_prompt() # Execute normal system interventions
                
            elif action == 'DECLINE_PROMPT' and getattr(self, 'prompt_active', False):
                print("[SYSTEM] Voice command declined prompt.")
                if getattr(self, 'prompt_reason', None) == 'VOICE_COMMAND_SUMMARY':
                    self.prompt_active = False
                    self.prompt_reason = None
                    self.voice_assistant.speak("Okay, skipping.")
                else:
                    self._decline_prompt()

        if self.tutorial.is_active:
            if key_char == 32:  
                self.tutorial.process_input(32, None)
                if not self.tutorial.is_active:
                    self.face_tracker.start_calibration()
            return 

        if key_char == ord('i') and self.gaze_reader.is_calibrated:
            self.inference_mode = not self.inference_mode
            
        # Dialog Inputs
        elif key_char in (ord('y'), ord('Y')) and self.dlg_active: 
            self.dlg_active = False
            self.dlg_para = None
        elif key_char in (ord('n'), ord('N')) and self.dlg_active:
            self.dlg_active = False
            self.dlg_para = None
            self.dlg_summary['text'] = None
            
        elif key_char in (ord('y'), ord('Y')) and getattr(self, 'prompt_active', False):
            if getattr(self, 'prompt_reason', None) == 'VOICE_COMMAND_SUMMARY':
                self.prompt_active = False
                self.prompt_reason = None
                self._trigger_deictic_summary()
            else:
                self._accept_prompt()
                
        elif key_char in (ord('n'), ord('N')) and getattr(self, 'prompt_active', False):
            if getattr(self, 'prompt_reason', None) == 'VOICE_COMMAND_SUMMARY':
                self.prompt_active = False
                self.prompt_reason = None
            else:
                self._decline_prompt()
                
        elif not self.dlg_active:
            if key_char == ord('j'):
                self.scroll_y = min(self.max_scroll, self.scroll_y + SCROLL_STEP)
                self.last_scroll_time = time.time()
            elif key_char == ord('k'):
                self.scroll_y = max(0, self.scroll_y - SCROLL_STEP)
                self.last_scroll_time = time.time()
            elif key_char == ord('n') and self.inference_mode:
                self.page = min(self.page + 1, self.pdf.n - 1)
                self.scroll_y = 0
                self.dialog_controller.reset_dwell()
                self.hov_buf.clear()
            elif key_char == ord('p') and self.inference_mode:
                self.page = max(self.page - 1, 0)
                self.scroll_y = 0
                self.dialog_controller.reset_dwell()
                self.hov_buf.clear()
            
            elif key_char == ord('r'):
                self.calib_index = 0
                self.sampling_active = False
                self.inference_mode = False 
                self.drift_mode = False
                self.drift_index = 0
                self.gaze_x, self.gaze_y = self.SCREEN_W//2, self.SCREEN_H//2
                self.gaze_reader.reset_calibration()
                self.heatmap.reset()
                self.dialog_controller.reset_dwell()
                self.hov_buf.clear()
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
