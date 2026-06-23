"""
Robust Eye Gaze Tracker + Plutchik Wheel + Local Epistemic Dashboard
====================================================================
1 Main Window Layout (1920x1080):
  - Left (1280x1080): Eye Tracker Sandbox
  - Top Right (640x540): Emotion Wheel (MediaPipe)
  - Bottom Right (640x540): Local Epistemic State (MediaPipe Heuristics)
"""

import cv2
import time
import numpy as np
import mediapipe as mp
import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from local_epistemic_tracker import LocalEpistemicTracker
from emotion_wheel import EmotionDetector, PlutchikWheel
from mouse_analyzer import MouseReadingAnalyzer
import requests
# =============================================================
# (EXISTING GAZE TRACKER CODE CONTINUES)
# =============================================================
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
        # 1. Predict next state
        self.kf.predict()
        
        # 2. Correct using current measurement (call only ONCE)
        measurement = np.array([[x], [y]], dtype=np.float32)
        corrected = self.kf.correct(measurement)
        
        # 3. Safely extract scalars using [row, column] indexing
        return float(corrected[0, 0]), float(corrected[1, 0])
    def reset(self): self._initialized = False

def build_feature(lx, ly, rx, ry, yaw, pitch, face_scale):
    yn = yaw / 30.0; pn = pitch / 20.0
    return np.array([lx, ly, rx, ry, yn, pn, face_scale, lx * yn, rx * yn, ly * pn, ry * pn], dtype=np.float64)

def _make_model(): return make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0))
model_x, model_y = _make_model(), _make_model()

def fit_calibration(features, targets, weights=None):
    X, Y = np.array(features), np.array(targets)
    sw = np.clip(np.array(weights, dtype=np.float64), 1e-6, None) / np.clip(np.array(weights, dtype=np.float64), 1e-6, None).max() if weights is not None and len(weights) == len(X) else None
    model_x.fit(X, Y[:, 0], gradientboostingregressor__sample_weight=sw); model_y.fit(X, Y[:, 1], gradientboostingregressor__sample_weight=sw)

class RangeStretcher:
    def __init__(self, margin=0.02): self.margin, self.fitted = margin, False
    def fit(self, features, targets):
        raw_x, raw_y = model_x.predict(np.array(features)), model_y.predict(np.array(features))
        self.x_min, self.x_max, self.y_min, self.y_max = raw_x.min(), raw_x.max(), raw_y.min(), raw_y.max()
        span_x, span_y = self.x_max - self.x_min, self.y_max - self.y_min
        self.x_min += span_x * self.margin; self.x_max -= span_x * self.margin; self.y_min += span_y * self.margin; self.y_max -= span_y * self.margin
        self.fitted = True
    def apply(self, raw_x, raw_y):
        if not self.fitted: return np.clip(raw_x, 0, 1), np.clip(raw_y, 0, 1)
        return float(np.clip((raw_x - self.x_min) / max(self.x_max - self.x_min, 1e-6), 0, 1)), float(np.clip((raw_y - self.y_min) / max(self.y_max - self.y_min, 1e-6), 0, 1))

stretcher = RangeStretcher()
def apply_calibration_stretched(feat): return stretcher.apply(float(model_x.predict(feat.reshape(1, -1))[0]), float(model_y.predict(feat.reshape(1, -1))[0]))

SANDBOX_W, SANDBOX_H, WHEEL_W, WHEEL_H = 1280, 1080, 640, 540

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

heatmap = GazeHeatmap()

_HEAD_3D = np.array([[0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0], [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]], dtype=np.float64)
_HEAD_IDX = [1, 152, 33, 263, 61, 291]

def get_head_pose(lm_norm, img_w, img_h):
    pts2d = np.array([(lm_norm[i].x * img_w, lm_norm[i].y * img_h) for i in _HEAD_IDX], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_HEAD_3D, pts2d, np.array([[float(img_w),0,img_w/2],[0,float(img_w),img_h/2],[0,0,1]], dtype=np.float64), np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return 0.0, 0.0, 0.0, np.eye(3)
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    return (np.degrees(np.arctan2(-R[2,0], sy)), np.degrees(np.arctan2( R[2,1], R[2,2])), np.degrees(np.arctan2( R[1,0], R[0,0])), R) if sy > 1e-6 else (0.0, np.degrees(np.arctan2(-R[2,0], sy)), np.degrees(np.arctan2(-R[0,1], R[1,1])), R)

def get_face_scale(lm_norm): return float(np.hypot(lm_norm[263].x - lm_norm[33].x, lm_norm[263].y - lm_norm[33].y))

def get_compensated_gaze(lm_px, R, img_w, img_h):
    nose = np.array(lm_px[1], dtype=np.float64)
    face_w = np.hypot(lm_px[263][0] - lm_px[33][0], lm_px[263][1] - lm_px[33][1]) + 1e-6
    def comp(indices):
        pt = np.mean([np.array(lm_px[i], dtype=np.float64) for i in indices], axis=0)
        vc = R.T @ np.array([(pt[0] - nose[0]) / face_w, (pt[1] - nose[1]) / face_w, 0.0])
        return float(vc[0]), float(vc[1])
    return *comp([474, 475, 476, 477]), *comp([469, 470, 471, 472])

def reject_outliers(samples, k=2.0):
    arr, median = np.array(samples), np.median(np.array(samples), axis=0)
    good = arr[np.all(np.abs(arr - median) <= k * (np.median(np.abs(arr - median), axis=0) + 1e-9), axis=1)]
    return good if len(good) >= max(5, len(arr)//4) else arr

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

IRIS_LEFT, EYE_LEFT_OUTER, EYE_LEFT_INNER, EYE_LEFT_TOP, EYE_LEFT_BOTTOM = [474, 475, 476, 477], 33, 133, [159, 160, 161], [145, 144, 163]
IRIS_RIGHT, EYE_RIGHT_OUTER, EYE_RIGHT_INNER, EYE_RIGHT_TOP, EYE_RIGHT_BOTTOM = [469, 470, 471, 472], 362, 263, [386, 387, 388], [374, 373, 390]
CALIB_POINTS_NORM = [(0.5, 0.5), (0.5, 0.3), (0.5, 0.7), (0.35, 0.5), (0.65, 0.5), (0.05, 0.05), (0.95, 0.05), (0.05, 0.95), (0.95, 0.95), (0.5, 0.01), (0.5, 0.99), (0.01, 0.5), (0.99, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
DRIFT_POINTS_NORM = [(0.5, 0.5), (0.05, 0.05), (0.95, 0.05), (0.05, 0.95), (0.95, 0.95)]
SAMPLES_NEEDED, DRIFT_SAMPLES_NEED = 15, 10
calib_index, calib_features, calib_targets, calib_weights, calib_done, sampling_active, sample_buffer, stale_feat_frames, MAX_STALE_FRAMES, drift_corrections, drift_mode, drift_index, drift_sampling, drift_sample_buf = 0, [], [], [], False, False, [], 0, 15, [], False, 0, False, []

def add_drift_correction(feat, tx, ty):
    drift_corrections.append((feat.copy(), [tx, ty]))
    if len(drift_corrections) > 30: drift_corrections.pop(0)
    fit_calibration(calib_features + [d[0] for d in drift_corrections], calib_targets + [d[1] for d in drift_corrections], calib_weights + [3.0 * max(calib_weights, default=1.0)] * len(drift_corrections))
    stretcher.fit(calib_features + [d[0] for d in drift_corrections], calib_targets + [d[1] for d in drift_corrections])

latest_landmarks, latest_blendshapes, new_data_available = None, None, False
landmark_lock = threading.Lock()

def result_callback(result, output_image, timestamp_ms):
    global latest_landmarks, latest_blendshapes, new_data_available
    with landmark_lock:
        latest_landmarks, latest_blendshapes = (result.face_landmarks[0], result.face_blendshapes[0]) if result.face_landmarks else (None, None)
        new_data_available = True

detector = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='face_landmarker.task'), num_faces=1,
    min_face_detection_confidence=0.5, min_tracking_confidence=0.5, output_face_blendshapes=True,
    running_mode=vision.RunningMode.LIVE_STREAM, result_callback=result_callback))

def get_eye_gaze_ratio(landmarks, iris_indices, outer, inner, top_ids, bottom_ids):
    iris_pts = [landmarks[i] for i in iris_indices]
    cx, cy = sum(p[0] for p in iris_pts) / len(iris_pts), sum(p[1] for p in iris_pts) / len(iris_pts)
    lx, ly, rx, ry = landmarks[outer][0], landmarks[outer][1], landmarks[inner][0], landmarks[inner][1]
    eye_w, eye_h = np.hypot(rx - lx, ry - ly), abs(sum(landmarks[i][1] for i in bottom_ids) / len(bottom_ids) - sum(landmarks[i][1] for i in top_ids) / len(top_ids))
    return ((cx - (lx + rx) / 2.0) / (eye_w + 1e-6), (cy - (sum(landmarks[i][1] for i in top_ids) / len(top_ids) + sum(landmarks[i][1] for i in bottom_ids) / len(bottom_ids)) / 2.0) / (eye_h + 1e-6)) if eye_h / (eye_w + 1e-6) >= 0.10 else (None, None)

def draw_target_dot(canvas, tx, ty, progress, label):
    cv2.ellipse(canvas, (tx, ty), (22, 22), -90, 0, int(360 * progress), (0, 255, 100), 3)
    cv2.circle(canvas, (tx, ty), 12, (0, 200, 80), -1)
    cv2.putText(canvas, label, ((SANDBOX_W - 440) // 2, SANDBOX_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

cv2.namedWindow('Gaze & Emotion Dashboard', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Gaze & Emotion Dashboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Starting camera…")
vs = WebcamVideoStream(src=0, width=1280, height=720).start()
time.sleep(1.0)

kalman = KalmanGaze()
emotion_detector = EmotionDetector()
emotion_wheel = PlutchikWheel(width=WHEEL_W, height=WHEEL_H)

# --- Initialize Local Epistemic Tracker ---
epistemic_tracker = LocalEpistemicTracker()
mouse_tracker = MouseReadingAnalyzer(window_size=3.0)
mouse_tracker.start()
lrx = lry = rrx = rry = head_yaw = head_pitch = face_scale_val = 0.0
current_feat, smooth_x, smooth_y = None, SANDBOX_W // 2, SANDBOX_H // 2
current_landmarks, current_blendshapes, last_frame_id, display_image = None, None, -1, None
training_in_progress = False



# 2. Init (same as before)
epistemic_tracker = LocalEpistemicTracker(window_frames=90, min_frames=20)






while True:
    current_frame_id = vs.frame_id
    sandbox = np.ones((SANDBOX_H, SANDBOX_W, 3), dtype=np.uint8) * 18

    if current_frame_id > last_frame_id:
        raw_frame = vs.read()
        if raw_frame is None: continue
        display_image = cv2.flip(raw_frame, 1)
      
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))
        detector.detect_async(mp_image, int(time.time() * 1000))
        last_frame_id = current_frame_id
    elif display_image is None: continue
    
    h, w = display_image.shape[:2]
    process_new_frame = False
    with landmark_lock:
        if new_data_available:
            current_landmarks, current_blendshapes, new_data_available, process_new_frame = latest_landmarks, latest_blendshapes, False, True

    # -- Emotion Wheel (Top Right) --
    wheel_canvas = emotion_wheel.render(emotion_detector.update(current_blendshapes) if process_new_frame else emotion_detector.scores)

    # -- Local Epistemic UI (Bottom Right) --
    if process_new_frame:
        epistemic_tracker.update(current_blendshapes)
    
    
    epistemic_canvas = np.zeros((WHEEL_H, WHEEL_W, 3), dtype=np.uint8)
    epistemic_tracker.render(epistemic_canvas)

    # -- Eye Tracker (Left) --
    if process_new_frame and current_landmarks:
        lm_px = [(lm.x * w, lm.y * h) for lm in current_landmarks]
        yaw, pitch, roll, R = get_head_pose(current_landmarks, w, h)
        fs = get_face_scale(current_landmarks)
        left_x, left_y = get_eye_gaze_ratio(lm_px, IRIS_LEFT, EYE_LEFT_OUTER, EYE_LEFT_INNER, EYE_LEFT_TOP, EYE_LEFT_BOTTOM)
        right_x, right_y = get_eye_gaze_ratio(lm_px, IRIS_RIGHT, EYE_RIGHT_OUTER, EYE_RIGHT_INNER, EYE_RIGHT_TOP, EYE_RIGHT_BOTTOM)

        eyes_open = (left_x is not None) and (right_x is not None)
        use_feat = eyes_open or (current_feat is not None and stale_feat_frames < MAX_STALE_FRAMES)

        if use_feat:
            if eyes_open:
                lrx, lry, rrx, rry = get_compensated_gaze(lm_px, R, w, h)
                head_yaw, head_pitch, face_scale_val = yaw, pitch, fs
                current_feat = build_feature(lrx, lry, rrx, rry, head_yaw, head_pitch, face_scale_val)
                stale_feat_frames = 0
            else:
                stale_feat_frames += 1

            if calib_done and current_feat is not None:
                sx, sy = apply_calibration_stretched(current_feat)
                kx, ky = kalman.update(int(sx * SANDBOX_W), int(sy * SANDBOX_H))
                smooth_x, smooth_y = int(np.clip(kx, 0, SANDBOX_W - 1)), int(np.clip(ky, 0, SANDBOX_H - 1))
                heatmap.update(smooth_x, smooth_y)

            if sampling_active and eyes_open: sample_buffer.append(current_feat.tolist())
            if drift_sampling and eyes_open: drift_sample_buf.append(current_feat.tolist())

    if not calib_done:
        if training_in_progress:
            cv2.putText(sandbox, "Training model, please wait…", (SANDBOX_W//2 - 280, SANDBOX_H//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 200, 255), 2)
        else:
            tx, ty = int(CALIB_POINTS_NORM[calib_index][0] * SANDBOX_W), int(CALIB_POINTS_NORM[calib_index][1] * SANDBOX_H)
            if sampling_active:
                draw_target_dot(sandbox, tx, ty, len(sample_buffer) / SAMPLES_NEEDED, f"Hold still…  {len(sample_buffer)}/{SAMPLES_NEEDED}")
                if len(sample_buffer) >= SAMPLES_NEEDED:
                    clean = reject_outliers(sample_buffer, k=2.0)
                    calib_features.append(clean.mean(axis=0)); calib_targets.append(list(CALIB_POINTS_NORM[calib_index])); calib_weights.append(1.0 / (clean.var(axis=0).mean() + 1e-6))
                    sample_buffer.clear(); sampling_active = False; calib_index += 1
                    if calib_index == len(CALIB_POINTS_NORM):
                        calib_done, training_in_progress = False, True
                        def _train():
                            global calib_done, training_in_progress
                            fit_calibration(calib_features, calib_targets, calib_weights); stretcher.fit(calib_features, calib_targets)
                            kalman.reset(); heatmap.reset()
                            calib_done, training_in_progress = True, False
                        threading.Thread(target=_train, daemon=True).start()
            else:
                pulse = int(10 + 6 * abs(np.sin(time.time() * 3)))
                cv2.circle(sandbox, (tx, ty), pulse + 6, (255, 255, 255), 2); cv2.circle(sandbox, (tx, ty), pulse, (0, 0, 220), -1)
                cv2.putText(sandbox, f"Look at dot ({calib_index+1}/{len(CALIB_POINTS_NORM)}) — press SPACE", ((SANDBOX_W - cv2.getTextSize(f"Look at dot ({calib_index+1}/{len(CALIB_POINTS_NORM)}) — press SPACE", cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0][0]) // 2, SANDBOX_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
            for i, (px, py) in enumerate(CALIB_POINTS_NORM):
                cv2.circle(sandbox, (int(px * SANDBOX_W), int(py * SANDBOX_H)), 8 if i < calib_index else 5 if i == calib_index else 6, (0, 200, 0) if i < calib_index else (255, 255, 255) if i == calib_index else (80, 80, 80), -1)
    else:
        heatmap.render(sandbox)
        cv2.line(sandbox, (smooth_x - 14, smooth_y), (smooth_x + 14, smooth_y), (255, 255, 255), 1)
        cv2.line(sandbox, (smooth_x, smooth_y - 14), (smooth_x, smooth_y + 14), (255, 255, 255), 1)
        cv2.circle(sandbox, (smooth_x, smooth_y), 4, (255, 255, 255), -1)
        cv2.putText(sandbox, "Tracking — 'd' drift  'r' recalibrate  'q' quit", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(sandbox, f"L({lrx:.3f},{lry:.3f})  R({rrx:.3f},{rry:.3f})  yaw={head_yaw:.1f}  pitch={head_pitch:.1f}  fixes={len(drift_corrections)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        if drift_mode:
            dx, dy = int(DRIFT_POINTS_NORM[drift_index][0] * SANDBOX_W), int(DRIFT_POINTS_NORM[drift_index][1] * SANDBOX_H)
            cv2.putText(sandbox, f"Drift correction ({drift_index+1}/{len(DRIFT_POINTS_NORM)}) — look at dot, press SPACE", (20, SANDBOX_H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 220, 50), 2)
            if drift_sampling:
                draw_target_dot(sandbox, dx, dy, len(drift_sample_buf) / DRIFT_SAMPLES_NEED, f"Hold still… {len(drift_sample_buf)}/{DRIFT_SAMPLES_NEED}")
                if len(drift_sample_buf) >= DRIFT_SAMPLES_NEED:
                    add_drift_correction(reject_outliers(drift_sample_buf, k=2.0).mean(axis=0), *DRIFT_POINTS_NORM[drift_index])
                    drift_sample_buf.clear(); drift_sampling = False; drift_index += 1
                    if drift_index >= len(DRIFT_POINTS_NORM): drift_mode, drift_index = False, 0; kalman.reset(); heatmap.reset()
            else:
                pulse = int(8 + 5 * abs(np.sin(time.time() * 3)))
                cv2.circle(sandbox, (dx, dy), pulse + 5, (255, 220, 50), 2); cv2.circle(sandbox, (dx, dy), pulse, (200, 160, 0), -1)

    # ==================================================================
    # COMPOSE MAIN DASHBOARD (1920 x 1080)
    # ==================================================================
    dashboard = np.ones((1080, 1920, 3), dtype=np.uint8) * 12
    dashboard[0:SANDBOX_H, 0:SANDBOX_W] = sandbox                     # Left
    dashboard[0:WHEEL_H, SANDBOX_W:1920] = wheel_canvas               # Top Right
    dashboard[WHEEL_H:1080, SANDBOX_W:1920] = epistemic_canvas        # Bottom Right
    mouse_score = mouse_tracker.get_score()
    # NEW: Draw Mouse Modality HUD over the bottom-left of the sandbox
    hud_x, hud_y = 30, SANDBOX_H - 100
    cv2.rectangle(dashboard, (hud_x - 10, hud_y - 40), (hud_x + 350, hud_y + 30), (30, 30, 30), -1)
    cv2.putText(dashboard, f"Mouse PAR Score: {mouse_score:.2f}", (hud_x, hud_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw a visual progress bar for the score
    bar_width = 300
    filled_width = int(bar_width * mouse_score)
    cv2.rectangle(dashboard, (hud_x, hud_y + 5), (hud_x + bar_width, hud_y + 15), (100, 100, 100), -1)
    cv2.rectangle(dashboard, (hud_x, hud_y + 5), (hud_x + filled_width, hud_y + 15), (0, 255, 100), -1)
    cv2.imshow('Gaze & Emotion Dashboard', dashboard)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('r'):
        calib_index = 0; calib_features.clear(); calib_targets.clear(); calib_weights.clear()
        calib_done = sampling_active = drift_mode = drift_sampling = False
        sample_buffer.clear(); drift_sample_buf.clear(); drift_corrections.clear()
        smooth_x, smooth_y = SANDBOX_W//2, SANDBOX_H//2
        kalman.reset(); heatmap.reset()
        cv2.setWindowProperty('Gaze & Emotion Dashboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    elif key == ord('d') and calib_done and not drift_mode:
        drift_mode = True; drift_index = 0; drift_sampling = False; drift_sample_buf.clear()
    elif key == ord(' '):
        if not calib_done and not sampling_active: sampling_active = True; sample_buffer.clear()
        elif calib_done and drift_mode and not drift_sampling: drift_sampling = True; drift_sample_buf.clear()

vs.stop()
detector.close()
mouse_tracker.stop()
cv2.destroyAllWindows()