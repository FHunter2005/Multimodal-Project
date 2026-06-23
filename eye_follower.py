"""
Refactored Multimodal Reader Helper Dashboard (With Calibration)
====================================================================
Architecture:
- Analyzers: Gaze, Emotion, Epistemic, Mouse
- Dialog Controller: Fuses data to trigger interventions
- App Loop: Read -> Process -> Fuse -> Render
"""

import cv2
import time
import numpy as np
import mediapipe as mp
import threading
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor

from local_epistemic_tracker import LocalEpistemicTracker
from emotion_wheel import EmotionDetector, PlutchikWheel
from mouse_analyzer import MouseReadingAnalyzer
from gaze_analyzer import GazeReadingAnalyzer
# =============================================================
# 1. CORE TRACKER HELPERS & CONSTANTS
# =============================================================
SANDBOX_W, SANDBOX_H, WHEEL_W, WHEEL_H = 1280, 1080, 640, 540

CALIB_POINTS_NORM = [(0.5, 0.5), (0.5, 0.3), (0.5, 0.7), (0.35, 0.5), (0.65, 0.5), 
                     (0.05, 0.05), (0.95, 0.05), (0.05, 0.95), (0.95, 0.95), 
                     (0.5, 0.01), (0.5, 0.99), (0.01, 0.5), (0.99, 0.5), 
                     (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
DRIFT_POINTS_NORM = [(0.5, 0.5), (0.05, 0.05), (0.95, 0.05), (0.05, 0.95), (0.95, 0.95)]
SAMPLES_NEEDED, DRIFT_SAMPLES_NEED = 15, 10

class KalmanGaze:
    def __init__(self, process_noise=1e-4, measurement_noise=8e-2, dt=1/30):
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
    def __init__(self, width=SANDBOX_W, height=SANDBOX_H, blob_sigma=55.0, decay=0.93, alpha=0.55):
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

def build_feature(lx, ly, rx, ry, yaw, pitch, face_scale):
    yn = yaw / 30.0; pn = pitch / 20.0
    return np.array([lx, ly, rx, ry, yn, pn, face_scale, lx * yn, rx * yn, ly * pn, ry * pn], dtype=np.float64)

def _make_model(): return make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0))

class RangeStretcher:
    def __init__(self, margin=0.02): self.margin, self.fitted = margin, False
    def fit(self, model_x, model_y, features):
        raw_x, raw_y = model_x.predict(np.array(features)), model_y.predict(np.array(features))
        self.x_min, self.x_max, self.y_min, self.y_max = raw_x.min(), raw_x.max(), raw_y.min(), raw_y.max()
        span_x, span_y = self.x_max - self.x_min, self.y_max - self.y_min
        self.x_min += span_x * self.margin; self.x_max -= span_x * self.margin; self.y_min += span_y * self.margin; self.y_max -= span_y * self.margin
        self.fitted = True
    def apply(self, raw_x, raw_y):
        if not self.fitted: return np.clip(raw_x, 0, 1), np.clip(raw_y, 0, 1)
        return float(np.clip((raw_x - self.x_min) / max(self.x_max - self.x_min, 1e-6), 0, 1)), float(np.clip((raw_y - self.y_min) / max(self.y_max - self.y_min, 1e-6), 0, 1))

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

_HEAD_3D = np.array([[0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0], [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]], dtype=np.float64)
_HEAD_IDX = [1, 152, 33, 263, 61, 291]
IRIS_LEFT, EYE_LEFT_OUTER, EYE_LEFT_INNER, EYE_LEFT_TOP, EYE_LEFT_BOTTOM = [474, 475, 476, 477], 33, 133, [159, 160, 161], [145, 144, 163]
IRIS_RIGHT, EYE_RIGHT_OUTER, EYE_RIGHT_INNER, EYE_RIGHT_TOP, EYE_RIGHT_BOTTOM = [469, 470, 471, 472], 362, 263, [386, 387, 388], [374, 373, 390]

def get_head_pose(lm_norm, img_w, img_h):
    pts2d = np.array([(lm_norm[i].x * img_w, lm_norm[i].y * img_h) for i in _HEAD_IDX], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_HEAD_3D, pts2d, np.array([[float(img_w),0,img_w/2],[0,float(img_w),img_h/2],[0,0,1]], dtype=np.float64), np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return 0.0, 0.0, 0.0, np.eye(3)
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    return (np.degrees(np.arctan2(-R[2,0], sy)), np.degrees(np.arctan2( R[2,1], R[2,2])), np.degrees(np.arctan2( R[1,0], R[0,0])), R) if sy > 1e-6 else (0.0, np.degrees(np.arctan2(-R[2,0], sy)), np.degrees(np.arctan2(-R[0,1], R[1,1])), R)

def get_face_scale(lm_norm): return float(np.hypot(lm_norm[263].x - lm_norm[33].x, lm_norm[263].y - lm_norm[33].y))

def get_compensated_gaze(lm_px, R, img_w, img_h):
    nose, face_w = np.array(lm_px[1], dtype=np.float64), np.hypot(lm_px[263][0] - lm_px[33][0], lm_px[263][1] - lm_px[33][1]) + 1e-6
    def comp(indices):
        pt = np.mean([np.array(lm_px[i], dtype=np.float64) for i in indices], axis=0)
        vc = R.T @ np.array([(pt[0] - nose[0]) / face_w, (pt[1] - nose[1]) / face_w, 0.0])
        return float(vc[0]), float(vc[1])
    return *comp(IRIS_LEFT), *comp(IRIS_RIGHT)

def get_eye_gaze_ratio(landmarks, iris_indices, outer, inner, top_ids, bottom_ids):
    iris_pts = [landmarks[i] for i in iris_indices]
    cx, cy = sum(p[0] for p in iris_pts) / len(iris_pts), sum(p[1] for p in iris_pts) / len(iris_pts)
    lx, ly, rx, ry = landmarks[outer][0], landmarks[outer][1], landmarks[inner][0], landmarks[inner][1]
    eye_w, eye_h = np.hypot(rx - lx, ry - ly), abs(sum(landmarks[i][1] for i in bottom_ids) / len(bottom_ids) - sum(landmarks[i][1] for i in top_ids) / len(top_ids))
    return ((cx - (lx + rx) / 2.0) / (eye_w + 1e-6), (cy - (sum(landmarks[i][1] for i in top_ids) / len(top_ids) + sum(landmarks[i][1] for i in bottom_ids) / len(bottom_ids)) / 2.0) / (eye_h + 1e-6)) if eye_h / (eye_w + 1e-6) >= 0.10 else (None, None)

def reject_outliers(samples, k=2.0):
    arr, median = np.array(samples), np.median(np.array(samples), axis=0)
    good = arr[np.all(np.abs(arr - median) <= k * (np.median(np.abs(arr - median), axis=0) + 1e-9), axis=1)]
    return good if len(good) >= max(5, len(arr)//4) else arr

def draw_target_dot(canvas, tx, ty, progress, label):
    cv2.ellipse(canvas, (tx, ty), (22, 22), -90, 0, int(360 * progress), (0, 255, 100), 3)
    cv2.circle(canvas, (tx, ty), 12, (0, 200, 80), -1)
    cv2.putText(canvas, label, ((SANDBOX_W - 440) // 2, SANDBOX_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)


# =============================================================
# 2. THE DIALOG CONTROLLER (Data Fusion Logic)
# =============================================================
class DialogController:
    def __init__(self):
        self.struggle_frames = 0
        self.help_active = False
        self.system_message = "Listening to User State..."

    def evaluate(self, mouse_score, gaze_score, epistemic_state, emotion_scores):
        
        # 1. Extract Face Confidence
        confusion = epistemic_state.get('confusion', 0.0) if isinstance(epistemic_state, dict) else 0.0
        anger = emotion_scores.get('anger', 0.0) if isinstance(emotion_scores, dict) else 0.0
        sadness = emotion_scores.get('sadness', 0.0) if isinstance(emotion_scores, dict) else 0.0
        
        plutchik_frustration = (anger * 0.70) + (sadness * 0.30)
        face_struggle = max(confusion, plutchik_frustration)

        # 2. Tri-Modal Data Fusion
        user_is_stuck = False
        
        # Rule 1: The Gaze Veto (Cognitive Load is too high)
        # If the eyes are constantly regressing or stuck, they are struggling, regardless of face/mouse.
        if gaze_score > 0.75:
            user_is_stuck = True

        # Rule 2: Gaze + Face Agreement
        # Mild cognitive struggle combined with mild facial confusion/frustration
        elif gaze_score > 0.40 and face_struggle > 0.40:
            user_is_stuck = True

        # Rule 3: The Mouse Backup
        # If gaze tracking is temporarily lost/noisy, use the Mouse + Face dual-modality
        elif mouse_score is not None:
            if face_struggle > 0.40 and mouse_score > 0.40:
                user_is_stuck = True
            elif mouse_score > 0.85: # Extreme mouse hesitation
                user_is_stuck = True

        # Rule 4: Face Only (If Gaze is low and Mouse is inactive)
        elif face_struggle > 0.70:
            user_is_stuck = True

        # 3. Temporal Smoothing
        if user_is_stuck:
            self.struggle_frames += 1
        else:
            self.struggle_frames = max(0, self.struggle_frames - 2)

        # 4. Trigger Logic
        TRIGGER_THRESHOLD = 30 
        
        if self.struggle_frames > TRIGGER_THRESHOLD and not self.help_active:
            self.help_active = True
            if plutchik_frustration > confusion:
                self.system_message = "INTERVENTION: High Frustration. Suggesting break."
            elif gaze_score > 0.6:
                self.system_message = "INTERVENTION: High Cognitive Load (Gaze). Simplifying text."
            else:
                self.system_message = "INTERVENTION: User Confused. Offering definitions."
                
        elif self.struggle_frames == 0 and self.help_active:
            self.help_active = False
            self.system_message = "User engaged. Normal reading."

        return {
            "help_active": self.help_active,
            "message": self.system_message,
            "struggle_level": min(1.0, self.struggle_frames / float(TRIGGER_THRESHOLD)), # ADDED BACK
            "gaze_score": gaze_score,
            "face_score": face_struggle
        }# =============================================================
# 3. MAIN APPLICATION CLASS (The Orchestrator)
# =============================================================
class ReaderHelperApp:
    def __init__(self):
        # Layout Config
        self.SANDBOX_W, self.SANDBOX_H = SANDBOX_W, SANDBOX_H
        self.WHEEL_W, self.WHEEL_H = WHEEL_W, WHEEL_H
        
        # Modules
        self.vs = WebcamVideoStream(src=0, width=1280, height=720).start()
        self.kalman = KalmanGaze()
        self.heatmap = GazeHeatmap()
        self.emotion_detector = EmotionDetector()
        self.emotion_wheel = PlutchikWheel(width=self.WHEEL_W, height=self.WHEEL_H)
        self.epistemic_tracker = LocalEpistemicTracker(window_frames=90, min_frames=20)
        self.mouse_tracker = MouseReadingAnalyzer()
        self.gaze_tracker = GazeReadingAnalyzer(window_size=5.0)
        self.dialog_controller = DialogController()
        self.current_gaze_score = 0.0
        # ML Models & Calibration State
        self.model_x, self.model_y = _make_model(), _make_model()
        self.stretcher = RangeStretcher()
        
        self.calib_index = 0
        self.calib_features, self.calib_targets, self.calib_weights = [], [], []
        self.calib_done = False
        self.training_in_progress = False
        self.sampling_active = False
        self.sample_buffer = []
        
        self.drift_mode = False
        self.drift_index = 0
        self.drift_sampling = False
        self.drift_sample_buf = []
        self.drift_corrections = []

        self.current_feat = None
        self.smooth_x, self.smooth_y = self.SANDBOX_W // 2, self.SANDBOX_H // 2
        self.stale_feat_frames, self.MAX_STALE_FRAMES = 0, 15
        self.lrx = self.lry = self.rrx = self.rry = self.head_yaw = self.head_pitch = 0.0

        # MediaPipe Setup
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
            if result.face_landmarks:
                self.latest_landmarks = result.face_landmarks[0]
                self.latest_blendshapes = result.face_blendshapes[0]
                self.new_data_available = True

    def _fit_calibration(self, features, targets, weights=None):
        X, Y = np.array(features), np.array(targets)
        sw = np.clip(np.array(weights, dtype=np.float64), 1e-6, None) / np.clip(np.array(weights, dtype=np.float64), 1e-6, None).max() if weights is not None and len(weights) == len(X) else None
        self.model_x.fit(X, Y[:, 0], gradientboostingregressor__sample_weight=sw)
        self.model_y.fit(X, Y[:, 1], gradientboostingregressor__sample_weight=sw)

    def _add_drift_correction(self, feat, tx, ty):
        self.drift_corrections.append((feat.copy(), [tx, ty]))
        if len(self.drift_corrections) > 30: self.drift_corrections.pop(0)
        self._fit_calibration(
            self.calib_features + [d[0] for d in self.drift_corrections], 
            self.calib_targets + [d[1] for d in self.drift_corrections], 
            self.calib_weights + [3.0 * max(self.calib_weights, default=1.0)] * len(self.drift_corrections))
        self.stretcher.fit(self.model_x, self.model_y, self.calib_features + [d[0] for d in self.drift_corrections])

    def run(self):
        print("Starting Reader Helper Architecture…")
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
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))
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

        # --- UPDATE TRACKERS ---
        mouse_score = self.mouse_tracker.get_score()
        emotion_state = self.emotion_detector.update(bs) if process_new_frame else self.emotion_detector.scores
        
        epistemic_state = {}
        if process_new_frame:
            self.epistemic_tracker.update(bs)
            epistemic_state = getattr(self.epistemic_tracker, 'current_state', {}) 

        # --- PROCESS EYE GAZE ---
        if process_new_frame and lm:
            lm_px = [(l.x * w, l.y * h) for l in lm]
            yaw, pitch, roll, R = get_head_pose(lm, w, h)
            fs = get_face_scale(lm)
            left_x, left_y = get_eye_gaze_ratio(lm_px, IRIS_LEFT, EYE_LEFT_OUTER, EYE_LEFT_INNER, EYE_LEFT_TOP, EYE_LEFT_BOTTOM)
            right_x, right_y = get_eye_gaze_ratio(lm_px, IRIS_RIGHT, EYE_RIGHT_OUTER, EYE_RIGHT_INNER, EYE_RIGHT_TOP, EYE_RIGHT_BOTTOM)

            eyes_open = (left_x is not None) and (right_x is not None)
            use_feat = eyes_open or (self.current_feat is not None and self.stale_feat_frames < self.MAX_STALE_FRAMES)

            if use_feat:
                if eyes_open:
                    self.lrx, self.lry, self.rrx, self.rry = get_compensated_gaze(lm_px, R, w, h)
                    self.head_yaw, self.head_pitch = yaw, pitch
                    self.current_feat = build_feature(self.lrx, self.lry, self.rrx, self.rry, yaw, pitch, fs)
                    self.stale_feat_frames = 0
                else:
                    self.stale_feat_frames += 1

                # Active Tracking
                if self.calib_done and self.current_feat is not None:
                    raw_x = float(self.model_x.predict(self.current_feat.reshape(1, -1))[0])
                    raw_y = float(self.model_y.predict(self.current_feat.reshape(1, -1))[0])
                    sx, sy = self.stretcher.apply(raw_x, raw_y)
                    kx, ky = self.kalman.update(int(sx * self.SANDBOX_W), int(sy * self.SANDBOX_H))
                    self.smooth_x, self.smooth_y = int(np.clip(kx, 0, self.SANDBOX_W - 1)), int(np.clip(ky, 0, self.SANDBOX_H - 1))
                    self.heatmap.update(self.smooth_x, self.smooth_y)
                    self.current_gaze_score = self.gaze_tracker.process_point(self.smooth_x, self.smooth_y)
                else:
                    # FIXED: Save to class variable
                    self.current_gaze_score = 0.0
                # Collect Samples
                if self.sampling_active and eyes_open: 
                    self.sample_buffer.append(self.current_feat.tolist())
                if self.drift_sampling and eyes_open: 
                    self.drift_sample_buf.append(self.current_feat.tolist())

        # --- DRAW CALIBRATION / TRACKING UI ---
        if not self.calib_done:
            self._draw_calibration_ui(sandbox)
        else:
            self._draw_tracking_ui(sandbox)

        # --- DATA FUSION & RENDERING ---
        system_action = self.dialog_controller.evaluate(mouse_score, self.current_gaze_score, epistemic_state, emotion_state)
        
        wheel_canvas = self.emotion_wheel.render(emotion_state)
        epistemic_canvas = np.zeros((self.WHEEL_H, self.WHEEL_W, 3), dtype=np.uint8)
        self.epistemic_tracker.render(epistemic_canvas)

        self._draw_fusion_hud(sandbox, mouse_score, system_action)

        dashboard = np.ones((1080, 1920, 3), dtype=np.uint8) * 12
        dashboard[0:self.SANDBOX_H, 0:self.SANDBOX_W] = sandbox                     
        dashboard[0:self.WHEEL_H, self.SANDBOX_W:1920] = wheel_canvas               
        dashboard[self.WHEEL_H:1080, self.SANDBOX_W:1920] = epistemic_canvas        

        cv2.imshow('Gaze & Emotion Dashboard', dashboard)

    def _draw_calibration_ui(self, sandbox):
        if self.training_in_progress:
            cv2.putText(sandbox, "Training model, please wait…", (self.SANDBOX_W//2 - 280, self.SANDBOX_H//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 200, 255), 2)
        else:
            tx, ty = int(CALIB_POINTS_NORM[self.calib_index][0] * self.SANDBOX_W), int(CALIB_POINTS_NORM[self.calib_index][1] * self.SANDBOX_H)
            if self.sampling_active:
                draw_target_dot(sandbox, tx, ty, len(self.sample_buffer) / SAMPLES_NEEDED, f"Hold still…  {len(self.sample_buffer)}/{SAMPLES_NEEDED}")
                if len(self.sample_buffer) >= SAMPLES_NEEDED:
                    clean = reject_outliers(self.sample_buffer, k=2.0)
                    self.calib_features.append(clean.mean(axis=0))
                    self.calib_targets.append(list(CALIB_POINTS_NORM[self.calib_index]))
                    self.calib_weights.append(1.0 / (clean.var(axis=0).mean() + 1e-6))
                    self.sample_buffer.clear()
                    self.sampling_active = False
                    self.calib_index += 1
                    
                    if self.calib_index == len(CALIB_POINTS_NORM):
                        self.calib_done, self.training_in_progress = False, True
                        def _train():
                            self._fit_calibration(self.calib_features, self.calib_targets, self.calib_weights)
                            self.stretcher.fit(self.model_x, self.model_y, self.calib_features)
                            self.kalman.reset()
                            self.heatmap.reset()
                            self.calib_done, self.training_in_progress = True, False
                        threading.Thread(target=_train, daemon=True).start()
            else:
                pulse = int(10 + 6 * abs(np.sin(time.time() * 3)))
                cv2.circle(sandbox, (tx, ty), pulse + 6, (255, 255, 255), 2)
                cv2.circle(sandbox, (tx, ty), pulse, (0, 0, 220), -1)
                cv2.putText(sandbox, f"Look at dot ({self.calib_index+1}/{len(CALIB_POINTS_NORM)}) — press SPACE", 
                            ((self.SANDBOX_W - cv2.getTextSize(f"Look at dot ({self.calib_index+1}/{len(CALIB_POINTS_NORM)}) — press SPACE", cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0][0]) // 2, self.SANDBOX_H // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
            
            for i, (px, py) in enumerate(CALIB_POINTS_NORM):
                cv2.circle(sandbox, (int(px * self.SANDBOX_W), int(py * self.SANDBOX_H)), 
                           8 if i < self.calib_index else 5 if i == self.calib_index else 6, 
                           (0, 200, 0) if i < self.calib_index else (255, 255, 255) if i == self.calib_index else (80, 80, 80), -1)

    def _draw_tracking_ui(self, sandbox):
        self.heatmap.render(sandbox)
        cv2.line(sandbox, (self.smooth_x - 14, self.smooth_y), (self.smooth_x + 14, self.smooth_y), (255, 255, 255), 1)
        cv2.line(sandbox, (self.smooth_x, self.smooth_y - 14), (self.smooth_x, self.smooth_y + 14), (255, 255, 255), 1)
        cv2.circle(sandbox, (self.smooth_x, self.smooth_y), 4, (255, 255, 255), -1)
        cv2.putText(sandbox, "Tracking — 'd' drift  'r' recalibrate  'q' quit", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(sandbox, f"L({self.lrx:.3f},{self.lry:.3f})  R({self.rrx:.3f},{self.rry:.3f})  yaw={self.head_yaw:.1f}  pitch={self.head_pitch:.1f}  fixes={len(self.drift_corrections)}", 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        if self.drift_mode:
            dx, dy = int(DRIFT_POINTS_NORM[self.drift_index][0] * self.SANDBOX_W), int(DRIFT_POINTS_NORM[self.drift_index][1] * self.SANDBOX_H)
            cv2.putText(sandbox, f"Drift correction ({self.drift_index+1}/{len(DRIFT_POINTS_NORM)}) — look at dot, press SPACE", 
                        (20, self.SANDBOX_H - 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 220, 50), 2)
            if self.drift_sampling:
                draw_target_dot(sandbox, dx, dy, len(self.drift_sample_buf) / DRIFT_SAMPLES_NEED, f"Hold still… {len(self.drift_sample_buf)}/{DRIFT_SAMPLES_NEED}")
                if len(self.drift_sample_buf) >= DRIFT_SAMPLES_NEED:
                    feat_mean = reject_outliers(self.drift_sample_buf, k=2.0).mean(axis=0)
                    self._add_drift_correction(feat_mean, *DRIFT_POINTS_NORM[self.drift_index])
                    self.drift_sample_buf.clear()
                    self.drift_sampling = False
                    self.drift_index += 1
                    if self.drift_index >= len(DRIFT_POINTS_NORM): 
                        self.drift_mode, self.drift_index = False, 0
                        self.kalman.reset()
                        self.heatmap.reset()
            else:
                pulse = int(8 + 5 * abs(np.sin(time.time() * 3)))
                cv2.circle(sandbox, (dx, dy), pulse + 5, (255, 220, 50), 2)
                cv2.circle(sandbox, (dx, dy), pulse, (200, 160, 0), -1)

    def _draw_fusion_hud(self, canvas, mouse_score, action_state):
        hud_x, hud_y = 30, self.SANDBOX_H - 120
        cv2.rectangle(canvas, (hud_x - 10, hud_y - 40), (hud_x + 550, hud_y + 100), (40, 40, 40), -1)
        
        color = (0, 100, 255) if action_state["help_active"] else (255, 255, 255)
        cv2.putText(canvas, f"SYSTEM: {action_state['message']}", (hud_x, hud_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # --- THE FIX: Safely check if mouse_score is None ---
        if mouse_score is not None:
            mouse_text = f"Mouse PAR Score: {mouse_score:.2f}"
            mouse_color = (200, 200, 200)
        else:
            mouse_text = "Mouse PAR Score: INACTIVE"
            mouse_color = (100, 100, 100) # Dim it out so the user knows it's off
            
        cv2.putText(canvas, mouse_text, (hud_x, hud_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouse_color, 1)
        # ----------------------------------------------------

        cv2.putText(canvas, f"Fusion Struggle Level: {action_state['struggle_level']:.2f}", (hud_x, hud_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    def _handle_input(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            self.running = False
        elif key == ord('r'):
            self.calib_index = 0
            self.calib_features.clear(); self.calib_targets.clear(); self.calib_weights.clear()
            self.calib_done = self.sampling_active = self.drift_mode = self.drift_sampling = False
            self.sample_buffer.clear(); self.drift_sample_buf.clear(); self.drift_corrections.clear()
            self.smooth_x, self.smooth_y = self.SANDBOX_W//2, self.SANDBOX_H//2
            self.kalman.reset(); self.heatmap.reset()
            cv2.setWindowProperty('Gaze & Emotion Dashboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif key == ord('d') and self.calib_done and not self.drift_mode:
            self.drift_mode = True
            self.drift_index = 0
            self.drift_sampling = False
            self.drift_sample_buf.clear()
        elif key == ord(' '):
            if not self.calib_done and not self.sampling_active: 
                self.sampling_active = True
                self.sample_buffer.clear()
            elif self.calib_done and self.drift_mode and not self.drift_sampling: 
                self.drift_sampling = True
                self.drift_sample_buf.clear()

    def cleanup(self):
        print("Cleaning up modules...")
        self.vs.stop()
        self.detector.close()
        self.mouse_tracker.stop()
        cv2.destroyAllWindows()

# =============================================================
# ENTRY POINT
# =============================================================
if __name__ == "__main__":
    app = ReaderHelperApp()
    app.run()