"""
Refactored Multimodal Reader Helper Dashboard
====================================================================
Architecture (PURE MATH TRACKING):
- Analyzers: Gaze, Emotion, Epistemic, Mouse
- Upgrades: 
  1. 13-Point "Reading Box" Polynomial Regression
  2. Pre-Regression Moving Average (Kills input noise)
  3. Fixation Deadzone (Locks cursor tight on words)
  4. Mathematical Head Pose Compensation
  5. Dynamic Blink Filtering
"""

import cv2
import time
import numpy as np
import mediapipe as mp
import threading
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from local_epistemic_tracker import LocalEpistemicTracker
from emotion_wheel import EmotionDetector, PlutchikWheel
from mouse_analyzer import MouseReadingAnalyzer
from gaze_analyzer import GazeReadingAnalyzer

# =============================================================
# 1. CORE TRACKER HELPERS & CONSTANTS
# =============================================================
SCREEN_W, SCREEN_H = 1920, 1080
SANDBOX_W, SANDBOX_H, WHEEL_W, WHEEL_H = 1280, 1080, 640, 540

# --- NEW: 13-Point "Reader's Grid" ---
CALIB_POINTS_NORM = [
    # Outer Edge Box
    (0.05, 0.05), (0.5, 0.05), (0.95, 0.05),
    (0.05, 0.5),               (0.95, 0.5),
    (0.05, 0.95), (0.5, 0.95), (0.95, 0.95),
    # Inner PDF Reading Box (Crucial for high accuracy reading)
    (0.3, 0.3), (0.7, 0.3),
    (0.3, 0.7), (0.7, 0.7),
    # Dead Center
    (0.5, 0.5)
]
# Lowered to 10 so calibration doesn't take forever with 13 points
SAMPLES_NEEDED = 10 

# --- NEW: Smoothing Tunings ---
  # Jump Kalman if movement > 12% of screen
FIXATION_DEADZONE = 0.015  # Freeze cursor completely if movement < 1.5% of screen (~28 pixels)

BLINK_EAR_THRESHOLD = 0.16

_HEAD_3D = np.array([
    [0.0, 0.0, 0.0],           # 1: Nose
    [0.0, -330.0, -65.0],      # 152: Chin
    [-225.0, 170.0, -135.0],   # 33: Left Eye Outer
    [225.0, 170.0, -135.0],    # 263: Right Eye Outer
    [-150.0, -150.0, -125.0],  # 61: Left Mouth
    [150.0, -150.0, -125.0]    # 291: Right Mouth
], dtype=np.float64)
_HEAD_IDX = [1, 152, 33, 263, 61, 291] 

IRIS_LEFT, EYE_LEFT_OUTER, EYE_LEFT_INNER, EYE_LEFT_TOP, EYE_LEFT_BOTTOM = [474, 475, 476, 477], 33, 133, [159, 160, 161], [145, 144, 163]
IRIS_RIGHT, EYE_RIGHT_OUTER, EYE_RIGHT_INNER, EYE_RIGHT_TOP, EYE_RIGHT_BOTTOM = [469, 470, 471, 472], 362, 263, [386, 387, 388], [374, 373, 390]

class KalmanGaze:
    # Increased measurement noise so Kalman drags a bit more smoothly
    def __init__(self, process_noise=1e-4, measurement_noise=1e-1, dt=1/30):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self._initialized = False
    def update(self, x, y):
        if not self._initialized:
            self.kf.statePre = np.array([[x],[y],[0],[0]], dtype=np.float32); self.kf.statePost = self.kf.statePre.copy(); self._initialized = True; return x, y
        self.kf.predict()
        corrected = self.kf.correct(np.array([[x], [y]], dtype=np.float32))
        return float(corrected[0, 0]), float(corrected[1, 0])
    def reset(self): self._initialized = False

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

class WebcamVideoStream:
    def __init__(self, src=0, width=1280, height=720):
        self.stream = cv2.VideoCapture(src); self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width); self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.stream.read(); self.stopped, self.frame_id = False, 0
    def start(self): threading.Thread(target=self.update, daemon=True).start(); return self
    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if grabbed: self.frame, self.frame_id = frame, self.frame_id + 1
    def read(self): return self.frame
    def stop(self): self.stopped = True; self.stream.release()

def get_head_rotation_matrix(lm, img_w, img_h):
    pts2d = np.array([(lm[i].x * img_w, lm[i].y * img_h) for i in _HEAD_IDX], dtype=np.float64)
    cam_matrix = np.array([[float(img_w), 0, img_w/2], [0, float(img_w), img_h/2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_HEAD_3D, pts2d, cam_matrix, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return np.eye(3)
    R, _ = cv2.Rodrigues(rvec)
    return R

def unrotate_face(lm, R, img_w, img_h):
    nose_3d = np.array([lm[1].x * img_w, lm[1].y * img_h, lm[1].z * img_w])
    unrotated_lm_2d = []
    for l in lm:
        pt = np.array([l.x * img_w, l.y * img_h, l.z * img_w]) - nose_3d
        u_pt = R.T @ pt  
        unrotated_lm_2d.append((u_pt[0] + nose_3d[0], u_pt[1] + nose_3d[1]))
    return unrotated_lm_2d

def get_eye_gaze_ratio(landmarks, iris_indices, outer, inner, top_ids, bottom_ids):
    iris_pts = [landmarks[i] for i in iris_indices]
    cx, cy = sum(p[0] for p in iris_pts) / len(iris_pts), sum(p[1] for p in iris_pts) / len(iris_pts)
    lx, rx = landmarks[outer][0], landmarks[inner][0]
    
    eye_w = np.hypot(rx - lx, landmarks[inner][1] - landmarks[outer][1])
    eye_h = abs(sum(landmarks[i][1] for i in bottom_ids) / len(bottom_ids) - sum(landmarks[i][1] for i in top_ids) / len(top_ids))
    
    ear = eye_h / (eye_w + 1e-6)
    ratio_x = (cx - lx) / (eye_w + 1e-6)
    ratio_y = (cy - sum(landmarks[i][1] for i in top_ids) / len(top_ids)) / (eye_h + 1e-6)
    
    return ratio_x, ratio_y, ear

def reject_outliers(samples, k=2.0):
    arr, median = np.array(samples), np.median(np.array(samples), axis=0)
    good = arr[np.all(np.abs(arr - median) <= k * (np.median(np.abs(arr - median), axis=0) + 1e-9), axis=1)]
    return good if len(good) >= max(5, len(arr)//4) else arr

def draw_target_dot(canvas, tx, ty, progress, label):
    cv2.ellipse(canvas, (tx, ty), (22, 22), -90, 0, int(360 * progress), (0, 255, 100), 3)
    cv2.circle(canvas, (tx, ty), 12, (0, 200, 80), -1)
    cv2.putText(canvas, label, ((SCREEN_W - 440) // 2, SCREEN_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

# =============================================================
# 2. THE DIALOG CONTROLLER (Data Fusion Logic)
# =============================================================
class DialogController:
    def __init__(self):
        self.struggle_frames = 0
        self.help_active = False
        self.system_message = "Listening to User State..."

    def evaluate(self, mouse_score, gaze_score, epistemic_state, emotion_scores):
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
        
        if self.struggle_frames > TRIGGER_THRESHOLD and not self.help_active:
            self.help_active = True
            if plutchik_frustration > confusion: self.system_message = "INTERVENTION: High Frustration. Take a deep breath or a short break."
            elif gaze_score > 0.6: self.system_message = "INTERVENTION: High Cognitive Load. Would you like a simpler summary?"
            else: self.system_message = "INTERVENTION: You seem confused. Here is a definition of the complex term."
                
        elif self.struggle_frames == 0 and self.help_active:
            self.help_active = False
            self.system_message = "User engaged. Normal reading."

        return {
            "help_active": self.help_active,
            "message": self.system_message,
            "struggle_level": min(1.0, self.struggle_frames / float(TRIGGER_THRESHOLD)),
            "gaze_score": gaze_score,
            "face_score": face_struggle
        }

    def reset_intervention(self):
        self.struggle_frames = 0
        self.help_active = False

# =============================================================
# 3. MAIN APPLICATION CLASS (The Orchestrator)
# =============================================================
class ReaderHelperApp:
    def __init__(self):
        self.SCREEN_W, self.SCREEN_H = SCREEN_W, SCREEN_H
        self.SANDBOX_W, self.SANDBOX_H = SANDBOX_W, SANDBOX_H
        self.WHEEL_W, self.WHEEL_H = WHEEL_W, WHEEL_H
        
        self.vs = WebcamVideoStream(src=0, width=1280, height=720).start()
        self.kalman = KalmanGaze(process_noise=1e-3, measurement_noise=5e-2)
        self.heatmap = GazeHeatmap()
        self.emotion_detector = EmotionDetector()
        self.emotion_wheel = PlutchikWheel(width=self.WHEEL_W, height=self.WHEEL_H)
        self.epistemic_tracker = LocalEpistemicTracker(window_frames=90, min_frames=20)
        self.mouse_tracker = MouseReadingAnalyzer()
        self.gaze_tracker = GazeReadingAnalyzer(window_size=5.0)
        self.dialog_controller = DialogController()
        self.current_gaze_score = 0.0
        
        self.inference_mode = False
        self.scroll_y = 0
        self.dummy_doc = self._create_dummy_document()
        
        self.calib_index = 0
        self.calib_features = []
        self.calib_done = False
        self.sampling_active = False
        self.sample_buffer = []
        self.blink_history = deque(maxlen=900)
        self.calib_coeffs_x = None
        self.calib_coeffs_y = None
        self.prev_norm_x = None
        self.prev_norm_y = None

        self.smooth_x, self.smooth_y = self.SCREEN_W // 2, self.SCREEN_H // 2
        
        # --- NEW: Pre-Regression Smoothing Queues ---
        self.raw_x_history = deque(maxlen=5) 
        self.raw_y_history = deque(maxlen=5)
        
        self.raw_eye_x = self.raw_eye_y = 0.5
        
        self.last_good_eye_x = 0.5
        self.last_good_eye_y = 0.5
        self.is_blinking = False

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

    def _create_dummy_document(self):
        doc = np.ones((3000, 1920, 3), dtype=np.uint8) * 230 
        cv2.rectangle(doc, (360, 50), (1560, 2950), (255, 255, 255), -1)
        cv2.putText(doc, "Chapter 4: The Epistemology of Multimodal AI", (420, 150), cv2.FONT_HERSHEY_DUPLEX, 1.3, (30, 30, 30), 2)
        cv2.line(doc, (420, 170), (1500, 170), (200, 200, 200), 2)
        y_offset = 240
        paragraph = "Multimodal systems process information from diverse sensory channels simultaneously. Unlike unimodal architectures, which might only process text or images in isolation, these systems attempt to achieve late-stage decision level fusion to determine complex internal states such as cognitive load, epistemic confusion, or emotional frustration. "
        for p in range(18):
            cv2.putText(doc, f"[{p+1}] " + paragraph[:80], (420, y_offset), cv2.FONT_HERSHEY_COMPLEX, 0.7, (50, 50, 50), 1)
            cv2.putText(doc, paragraph[80:160], (420, y_offset + 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (50, 50, 50), 1)
            cv2.putText(doc, paragraph[160:], (420, y_offset + 70), cv2.FONT_HERSHEY_COMPLEX, 0.7, (50, 50, 50), 1)
            y_offset += 150
        return doc

    def _mp_callback(self, result, output_image, timestamp_ms):
        with self.landmark_lock:
            if result.face_landmarks:
                self.latest_landmarks = result.face_landmarks[0]
                self.latest_blendshapes = result.face_blendshapes[0]
                self.new_data_available = True

    def _fit_polynomial_calibration(self):
        X = np.array(self.calib_features)
        Y = np.array(CALIB_POINTS_NORM)
        
        A = np.column_stack([
            np.ones(len(X)), X[:, 0], X[:, 1], X[:, 0]**2, X[:, 1]**2, X[:, 0] * X[:, 1]
        ])
        
        self.calib_coeffs_x, _, _, _ = np.linalg.lstsq(A, Y[:, 0], rcond=None)
        self.calib_coeffs_y, _, _, _ = np.linalg.lstsq(A, Y[:, 1], rcond=None)

        self.calib_done = True
        self.kalman.reset()
        self.heatmap.reset()
        print("Calibration Complete: 13-Point Reader Curve Fitted.")

    def run(self):
        print("Starting Smooth Math Reader Helper Baseline…")
        time.sleep(1.0)
        cv2.namedWindow('Gaze & Emotion Dashboard', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Gaze & Emotion Dashboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while self.running:
            self._process_camera_feed()
            self._update_and_fuse()
            self._handle_input()
            
        self.cleanup()

    def _process_camera_feed(self):
        current_frame_id = self.vs.frame_id
        if current_frame_id > self.last_frame_id:
            raw_frame = self.vs.read()
            if raw_frame is not None:
                self.display_image = cv2.flip(raw_frame, 1)
                
                # Bypass heavy CLAHE processing to instantly restore FPS
                image_rgb = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                self.detector.detect_async(mp_image, int(time.time() * 1000))
                self.last_frame_id = current_frame_id

    def _update_and_fuse(self):
        if self.display_image is None: return

        h, w = self.display_image.shape[:2]
        process_new_frame = False
        sandbox = np.ones((self.SANDBOX_H, self.SANDBOX_W, 3), dtype=np.uint8) * 18
        
        with self.landmark_lock:
            if self.new_data_available:
                lm, bs = self.latest_landmarks, self.latest_blendshapes
                self.new_data_available, process_new_frame = False, True

        mouse_score = self.mouse_tracker.get_score()
        emotion_state = self.emotion_detector.update(bs) if process_new_frame else self.emotion_detector.scores
        
        epistemic_state = {}
        if process_new_frame:
            self.epistemic_tracker.update(bs)
            epistemic_state = getattr(self.epistemic_tracker, 'current_state', {}) 

        if process_new_frame and lm:
            R = get_head_rotation_matrix(lm, w, h)
            unrotated_lm_px = unrotate_face(lm, R, w, h)
            
            left_x, left_y, left_ear = get_eye_gaze_ratio(unrotated_lm_px, IRIS_LEFT, EYE_LEFT_OUTER, EYE_LEFT_INNER, EYE_LEFT_TOP, EYE_LEFT_BOTTOM)
            right_x, right_y, right_ear = get_eye_gaze_ratio(unrotated_lm_px, IRIS_RIGHT, EYE_RIGHT_OUTER, EYE_RIGHT_INNER, EYE_RIGHT_TOP, EYE_RIGHT_BOTTOM)

            if left_x is not None and right_x is not None:
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Pre-regression averaging to kill MediaPipe micro-jitter
                instant_raw_x = (left_x + right_x) / 2.0
                instant_raw_y = (left_y + right_y) / 2.0
                
                if avg_ear > BLINK_EAR_THRESHOLD:
                    self.is_blinking = False
                    self.blink_history.append(0)
                    self.raw_x_history.append(instant_raw_x)
                    self.raw_y_history.append(instant_raw_y)
                    
                    self.raw_eye_x = sum(self.raw_x_history) / len(self.raw_x_history)
                    self.raw_eye_y = sum(self.raw_y_history) / len(self.raw_y_history)
                    
                    self.last_good_eye_x = self.raw_eye_x
                    self.last_good_eye_y = self.raw_eye_y
                else:
                    self.is_blinking = True
                    self.blink_history.append(1)
                    self.raw_eye_x = self.last_good_eye_x
                    self.raw_eye_y = self.last_good_eye_y

                if len(self.blink_history) > 100:
                    perclos_score = sum(self.blink_history) / len(self.blink_history)
                    
                    if perclos_score > 0.15: # Eyes closed more than 15% of the time
                        self.dialog_controller.system_message = "INTERVENTION: High Fatigue Detected. Consider a screen break."
                        self.dialog_controller.help_active = True
                if self.sampling_active and not self.is_blinking: 
                    self.sample_buffer.append([self.raw_eye_x, self.raw_eye_y])
                if self.calib_done:
                    # 1. Run raw eye ratio through Kalman filter
                    smooth_raw_x, smooth_raw_y = self.kalman.update(self.raw_eye_x, self.raw_eye_y)

                    # 2. Build feature matrix using the smoothed, stable input
                    feat = np.array([
                        1, 
                        smooth_raw_x, 
                        smooth_raw_y, 
                        smooth_raw_x**2, 
                        smooth_raw_y**2, 
                        smooth_raw_x * smooth_raw_y
                    ])
                    
                    # 3. Calculate Normalized Screen Coordinates
                    norm_x = float(np.dot(feat, self.calib_coeffs_x))
                    norm_y = float(np.dot(feat, self.calib_coeffs_y))
                    
                    # 4. Clamp for safety
                    norm_x = float(np.clip(norm_x, 0.0, 1.0))
                    norm_y = float(np.clip(norm_y, 0.0, 1.0))
                    
                    # 5. Apply Fixation Deadzone (The "Lock" feeling)
                    # If movement is tiny, ignore it to keep cursor rock-solid
                    if self.prev_norm_x is not None and not self.is_blinking:
                        dist = np.hypot(norm_x - self.prev_norm_x, norm_y - self.prev_norm_y)
                        if dist < FIXATION_DEADZONE:
                            norm_x, norm_y = self.prev_norm_x, self.prev_norm_y
                    
                    self.prev_norm_x, self.prev_norm_y = norm_x, norm_y

                    # 6. Smooth update for screen position
                    self.smooth_x = int(norm_x * self.SCREEN_W)
                    self.smooth_y = int(norm_y * self.SCREEN_H)
                    
                    self.heatmap.update(self.smooth_x, self.smooth_y)
                    self.current_gaze_score = self.gaze_tracker.process_point(self.smooth_x, self.smooth_y)
            else:
                self.current_gaze_score = 0.0

        system_action = self.dialog_controller.evaluate(mouse_score, self.current_gaze_score, epistemic_state, emotion_state)

        # =============================================================
        # RENDERING
        # =============================================================
        if self.inference_mode and self.calib_done:
            max_scroll = self.dummy_doc.shape[0] - 1080
            self.scroll_y = max(0, min(self.scroll_y, max_scroll))
            view = self.dummy_doc[self.scroll_y : self.scroll_y + 1080, :, :].copy()

            if system_action['help_active']:
                cx, cy = self.SCREEN_W // 2, self.SCREEN_H // 2
                cv2.rectangle(view, (cx - 380, cy - 100), (cx + 380, cy + 120), (150, 150, 150), 4)
                cv2.rectangle(view, (cx - 380, cy - 100), (cx + 380, cy + 120), (250, 250, 255), -1)
                cv2.putText(view, "READER HELPER", (cx - 350, cy - 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 100, 200), 2)
                cv2.line(view, (cx - 350, cy - 35), (cx + 350, cy - 35), (200, 200, 200), 2)
                cv2.putText(view, system_action['message'], (cx - 350, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
                cv2.putText(view, "[Press 'C' to clear and continue reading]", (cx - 150, cy + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            cv2.putText(view, "INFERENCE MODE ACTIVE", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
            cv2.putText(view, "[I] Return to Dashboard   [W]/[S] Scroll Document", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            if self.is_blinking:
                cv2.putText(view, "BLINK DETECTED - COORDS FROZEN", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.circle(view, (self.smooth_x, self.smooth_y), 5, (0, 0, 255), -1)

            cv2.imshow('Gaze & Emotion Dashboard', view)

        else:
            if not self.calib_done:
                dashboard = np.ones((self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8) * 18
                self._draw_calibration_ui(dashboard, self.SCREEN_W, self.SCREEN_H)
            else:
                wheel_canvas = self.emotion_wheel.render(emotion_state)
                epistemic_canvas = np.zeros((self.WHEEL_H, self.WHEEL_W, 3), dtype=np.uint8)
                self.epistemic_tracker.render(epistemic_canvas)

                dashboard = np.ones((self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8) * 12
                dashboard[0:self.SANDBOX_H, 0:self.SANDBOX_W] = sandbox                     
                dashboard[0:self.WHEEL_H, self.SANDBOX_W:self.SCREEN_W] = wheel_canvas               
                dashboard[self.WHEEL_H:self.SCREEN_H, self.SANDBOX_W:self.SCREEN_W] = epistemic_canvas        
                
                self._draw_tracking_ui(dashboard, self.SCREEN_W, self.SCREEN_H)
                self._draw_fusion_hud(dashboard, mouse_score, system_action)
                cv2.putText(dashboard, "Press [I] to enter INFERENCE MODE (Reader View)", (1320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Gaze & Emotion Dashboard', dashboard)

    def _draw_calibration_ui(self, canvas, w, h):
        tx, ty = int(CALIB_POINTS_NORM[self.calib_index][0] * w), int(CALIB_POINTS_NORM[self.calib_index][1] * h)
        if self.sampling_active:
            draw_target_dot(canvas, tx, ty, len(self.sample_buffer) / SAMPLES_NEEDED, f"Hold still…  {len(self.sample_buffer)}/{SAMPLES_NEEDED}")
            if len(self.sample_buffer) >= SAMPLES_NEEDED:
                clean = reject_outliers(self.sample_buffer, k=2.0)
                self.calib_features.append(clean.mean(axis=0))
                self.sample_buffer.clear()
                self.sampling_active = False
                self.calib_index += 1
                
                if self.calib_index == len(CALIB_POINTS_NORM):
                    self._fit_polynomial_calibration()
        else:
            pulse = int(10 + 6 * abs(np.sin(time.time() * 3)))
            cv2.circle(canvas, (tx, ty), pulse + 6, (255, 255, 255), 2)
            cv2.circle(canvas, (tx, ty), pulse, (0, 0, 220), -1)
            
            text = f"Look at dot ({self.calib_index+1}/{len(CALIB_POINTS_NORM)}) — press SPACE"
            text_w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0][0]
            cv2.putText(canvas, text, ((w - text_w) // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
        
        for i, (px, py) in enumerate(CALIB_POINTS_NORM):
            cv2.circle(canvas, (int(px * w), int(py * h)), 
                       8 if i < self.calib_index else 5 if i == self.calib_index else 6, 
                       (0, 200, 0) if i < self.calib_index else (255, 255, 255) if i == self.calib_index else (80, 80, 80), -1)

    def _draw_tracking_ui(self, canvas, w, h):
        self.heatmap.render(canvas)
        cv2.line(canvas, (self.smooth_x - 14, self.smooth_y), (self.smooth_x + 14, self.smooth_y), (255, 255, 255), 1)
        cv2.line(canvas, (self.smooth_x, self.smooth_y - 14), (self.smooth_x, self.smooth_y + 14), (255, 255, 255), 1)
        cv2.circle(canvas, (self.smooth_x, self.smooth_y), 4, (255, 255, 255), -1)
        
        cv2.putText(canvas, "Tracking Base — 'r' recalibrate  'q' quit", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        status_text = f"Smoothed Ratios: X({self.raw_eye_x:.3f}) Y({self.raw_eye_y:.3f})"
        if self.is_blinking: status_text += " [BLINKING]"
        cv2.putText(canvas, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255) if self.is_blinking else (200,200,200), 1)

    def _draw_fusion_hud(self, canvas, mouse_score, action_state):
        hud_x, hud_y = 30, self.SANDBOX_H - 120
        cv2.rectangle(canvas, (hud_x - 10, hud_y - 40), (hud_x + 550, hud_y + 100), (40, 40, 40), -1)
        
        color = (0, 100, 255) if action_state["help_active"] else (255, 255, 255)
        cv2.putText(canvas, f"SYSTEM: {action_state['message']}", (hud_x, hud_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if mouse_score is not None:
            mouse_text = f"Mouse PAR Score: {mouse_score:.2f}"
            mouse_color = (200, 200, 200)
        else:
            mouse_text = "Mouse PAR Score: INACTIVE"
            mouse_color = (100, 100, 100)
            
        cv2.putText(canvas, mouse_text, (hud_x, hud_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouse_color, 1)
        cv2.putText(canvas, f"Fusion Struggle Level: {action_state['struggle_level']:.2f}", (hud_x, hud_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    def _handle_input(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            self.running = False
        elif key == ord('i') and self.calib_done:
            self.inference_mode = not self.inference_mode
        elif key == ord('s') and self.inference_mode:
            self.scroll_y += 60
        elif key == ord('w') and self.inference_mode:
            self.scroll_y -= 60
        elif key == ord('c') and self.dialog_controller.help_active:
            self.dialog_controller.reset_intervention()
        elif key == ord('r'):
            self.calib_index = 0
            self.calib_features.clear()
            self.calib_done = self.sampling_active = False
            self.inference_mode = False 
            self.sample_buffer.clear()
            
            self.raw_x_history.clear()
            self.raw_y_history.clear()
            
            self.smooth_x, self.smooth_y = self.SCREEN_W//2, self.SCREEN_H//2
            self.prev_norm_x = self.prev_norm_y = None
            self.last_good_eye_x = self.last_good_eye_y = 0.5
            self.kalman.reset(); self.heatmap.reset()
            cv2.setWindowProperty('Gaze & Emotion Dashboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif key == ord(' ') and not self.calib_done and not self.sampling_active: 
            self.sampling_active = True
            self.sample_buffer.clear()

    def cleanup(self):
        print("Cleaning up modules...")
        self.vs.stop()
        self.detector.close()
        self.mouse_tracker.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ReaderHelperApp()
    app.run()