"""
Robust Eye Gaze Tracker
=======================
Key improvements over the baseline:
  1. Head pose (yaw + pitch from solvePnP) added as SVR features
  2. Per-eye iris ratios kept separate (4 values instead of averaged 2)
  3. Face scale (inter-ocular distance) as a distance-normalisation feature
  4. Calibration outlier rejection via Median Absolute Deviation
  5. Kalman filter (velocity-aware) replaces dual-alpha EMA
  6. Polynomial feature cross-terms (iris × head-pose interactions)
  7. Per-point confidence weighting during calibration fit
  8. Online drift correction — press 'd' during tracking to recorrect
"""

import cv2
import time
import numpy as np
import mediapipe as mp
import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# =============================================================
# Kalman Filter
# =============================================================
class KalmanGaze:
    """
    2-D Kalman filter with a constant-velocity model.
    State  : [x, y, vx, vy]
    Measure: [x, y]
    """
    def __init__(self, process_noise: float = 1e-4,
                 measurement_noise: float = 8e-2, dt: float = 1 / 30):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0],
             [0, 1,  0, dt],
             [0, 0,  1,  0],
             [0, 0,  0,  1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self._initialized = False

    def update(self, x: float, y: float):
        if not self._initialized:
            self.kf.statePre  = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.kf.statePost = self.kf.statePre.copy()
            self._initialized = True
            return x, y
        self.kf.predict()
        corrected = self.kf.correct(np.array([[x], [y]], dtype=np.float32))
        return float(corrected[0]), float(corrected[1])

    def reset(self):
        self._initialized = False


# =============================================================
# Feature Engineering
# =============================================================
def build_feature(lx, ly, rx, ry, yaw, pitch, face_scale):
    """
    11-D raw feature vector.
    The pipeline's PolynomialFeatures then expands this further,
    but we already bake in the most meaningful cross-terms explicitly
    so the SVR kernel sees them directly at their natural scale.
    """
    yn = yaw   / 30.0   # normalise: ±30° wide head turn
    pn = pitch / 20.0   # normalise: ±20° wide nod
    return np.array([
        lx, ly, rx, ry,   # per-eye iris ratios
        yn, pn,            # head pose
        face_scale,        # inter-ocular / img-width (distance proxy)
        lx * yn,           # iris-x × yaw  (most important cross-term)
        rx * yn,
        ly * pn,           # iris-y × pitch
        ry * pn,
    ], dtype=np.float64)


# =============================================================
# ML Models
# =============================================================
def _make_model():
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        SVR(C=5.0, epsilon=0.01, kernel='rbf', gamma='scale'),
    )

model_x = _make_model()
model_y = _make_model()


def fit_calibration(features, targets, weights=None):
    """
    features : list/array of 11-D feature vectors
    targets  : list/array of (norm_x, norm_y) pairs
    weights  : optional list of per-point confidence scalars
    """
    X = np.array(features)   # (N, 11)
    Y = np.array(targets)    # (N, 2)

    if weights is not None and len(weights) == len(X):
        W = np.array(weights, dtype=np.float64)
        W = np.clip(W, 1e-6, None)
        # SVR has no sample_weight; oversample proportionally instead
        counts = np.round(W / W.min() * 3).astype(int)
        X = np.repeat(X, counts, axis=0)
        Y = np.repeat(Y, counts, axis=0)

    model_x.fit(X, Y[:, 0])
    model_y.fit(X, Y[:, 1])


def apply_calibration(feat: np.ndarray):
    pred = feat.reshape(1, -1)
    sx = float(model_x.predict(pred)[0])
    sy = float(model_y.predict(pred)[0])
    return np.clip(sx, 0.0, 1.0), np.clip(sy, 0.0, 1.0)


# =============================================================
# Head Pose via solvePnP
# =============================================================
_HEAD_3D = np.array([
    [  0.0,    0.0,    0.0],   # 1   nose tip
    [  0.0, -330.0,  -65.0],   # 152 chin
    [-225.0,  170.0, -135.0],  # 33  left  eye outer
    [ 225.0,  170.0, -135.0],  # 263 right eye outer
    [-150.0, -150.0, -125.0],  # 61  left  mouth corner
    [ 150.0, -150.0, -125.0],  # 291 right mouth corner
], dtype=np.float64)
_HEAD_IDX = [1, 152, 33, 263, 61, 291]


def get_head_pose(lm_norm, img_w: int, img_h: int):
    pts2d = np.array(
        [(lm_norm[i].x * img_w, lm_norm[i].y * img_h) for i in _HEAD_IDX],
        dtype=np.float64)
    fl = float(img_w)
    cam_mat = np.array([[fl, 0, img_w / 2],
                        [0, fl, img_h / 2],
                        [0,  0,          1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        _HEAD_3D, pts2d, cam_mat, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = np.degrees(np.arctan2( R[2, 1], R[2, 2]))
        roll  = np.degrees(np.arctan2( R[1, 0], R[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = 0.0
        roll  = np.degrees(np.arctan2(-R[0, 1], R[1, 1]))
    return yaw, pitch, roll


def get_face_scale(lm_norm) -> float:
    l, r = lm_norm[33], lm_norm[263]
    return float(np.hypot(r.x - l.x, r.y - l.y))


# =============================================================
# Calibration Outlier Rejection
# =============================================================
def reject_outliers(samples: list, k: float = 2.0) -> np.ndarray:
    arr    = np.array(samples)
    median = np.median(arr, axis=0)
    mad    = np.median(np.abs(arr - median), axis=0) + 1e-9
    mask   = np.all(np.abs(arr - median) <= k * mad, axis=1)
    good   = arr[mask]
    return good if len(good) >= max(5, len(arr) // 4) else arr


# =============================================================
# Threaded Camera
# =============================================================
class WebcamVideoStream:
    def __init__(self, src=0, width=1280, height=720):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.stream.read()
        self.stopped  = False
        self.frame_id = 0

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if grabbed:
                self.frame    = frame
                self.frame_id += 1

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


# =============================================================
# Constants & Landmark Indices
# =============================================================
IRIS_LEFT        = [474, 475, 476, 477]
EYE_LEFT_OUTER   = 33
EYE_LEFT_INNER   = 133
EYE_LEFT_TOP     = [159, 160, 161]
EYE_LEFT_BOTTOM  = [145, 144, 163]

IRIS_RIGHT       = [469, 470, 471, 472]
EYE_RIGHT_OUTER  = 362
EYE_RIGHT_INNER  = 263
EYE_RIGHT_TOP    = [386, 387, 388]
EYE_RIGHT_BOTTOM = [374, 373, 390]

SCREEN_WIDTH  = 1920
SCREEN_HEIGHT = 1080

CALIB_POINTS_NORM = [
    (0.50, 0.50),
    (0.50, 0.30), (0.50, 0.70),
    (0.35, 0.50), (0.65, 0.50),
    (0.05, 0.05), (0.95, 0.05),
    (0.05, 0.95), (0.95, 0.95),
    (0.50, 0.01), (0.50, 0.99),
    (0.01, 0.50), (0.99, 0.50),
    (0.25, 0.25), (0.75, 0.25),
    (0.25, 0.75), (0.75, 0.75),
]

# Drift correction uses just 5 points: centre + 4 corners.
# Enough to capture the main affine drift without a long re-calibration.
DRIFT_POINTS_NORM = [
    (0.50, 0.50),
    (0.05, 0.05), (0.95, 0.05),
    (0.05, 0.95), (0.95, 0.95),
]

SAMPLES_NEEDED = 45   # per calibration point

# =============================================================
# Calibration state  (defined ONCE — not inside the loop)
# =============================================================
calib_index    = 0
calib_features = []   # averaged 11-D feature per point
calib_targets  = []   # mirrors CALIB_POINTS_NORM order
calib_weights  = []   # per-point confidence scalars   ← fixed: was inside loop
calib_done     = False

sampling_active = False
sample_buffer   = []


# =============================================================
# Online Drift Correction  (NEW)
# =============================================================
drift_corrections = []   # list of (feat_11d, [norm_x, norm_y])

# How many drift samples before we trigger a retrain
DRIFT_RETRAIN_EVERY = 1   # retrain after every new drift point


def add_drift_correction(feat: np.ndarray, true_x_norm: float, true_y_norm: float):
    """
    Record one ground-truth fixation and retrain both models.

    We keep the original calibration data plus all drift corrections,
    then re-fit.  Drift corrections are up-weighted (3×) so recent
    positional ground truth dominates over stale calibration data.
    """
    drift_corrections.append((feat.copy(), [true_x_norm, true_y_norm]))

    # Keep a bounded window so very old drift data doesn't accumulate
    max_drift = 30
    if len(drift_corrections) > max_drift:
        drift_corrections.pop(0)

    # Build combined dataset
    all_feats   = calib_features + [d[0] for d in drift_corrections]
    all_targets = calib_targets  + [d[1] for d in drift_corrections]

    # Drift corrections get higher weight than original calib points
    drift_w = 3.0
    all_weights = calib_weights + [
        drift_w * max(calib_weights, default=1.0) for _ in drift_corrections
    ]

    fit_calibration(all_feats, all_targets, all_weights)
    print(f"[Drift] Retrained with {len(all_feats)} points "
          f"({len(drift_corrections)} drift fixes).")


# Drift-correction session state
drift_mode         = False   # True while collecting drift points
drift_index        = 0       # which DRIFT_POINTS_NORM we're on
drift_sampling     = False   # True while collecting frames for current point
drift_sample_buf   = []      # raw feature vectors for current drift point
DRIFT_SAMPLES_NEED = 30      # fewer samples than full calib — it's a quick fix


# =============================================================
# MediaPipe Async Setup
# =============================================================
latest_landmarks   = None
landmark_lock      = threading.Lock()
new_data_available = False


def result_callback(result, output_image, timestamp_ms):
    global latest_landmarks, new_data_available
    with landmark_lock:
        latest_landmarks   = result.face_landmarks[0] if result.face_landmarks else None
        new_data_available = True


base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=result_callback,
)
detector = vision.FaceLandmarker.create_from_options(options)


# =============================================================
# Eye Gaze Ratio
# =============================================================
def get_eye_gaze_ratio(landmarks, iris_indices, outer, inner, top_ids, bottom_ids):
    iris_pts = [landmarks[i] for i in iris_indices]
    cx = sum(p[0] for p in iris_pts) / len(iris_pts)
    cy = sum(p[1] for p in iris_pts) / len(iris_pts)

    lx, ly = landmarks[outer]
    rx, ry = landmarks[inner]
    eye_w  = np.hypot(rx - lx, ry - ly)

    top_y  = sum(landmarks[i][1] for i in top_ids)    / len(top_ids)
    bot_y  = sum(landmarks[i][1] for i in bottom_ids) / len(bottom_ids)
    eye_h  = abs(bot_y - top_y)

    ear = eye_h / (eye_w + 1e-6)
    if ear < 0.18:   # blink → discard
        return None, None

    eye_cx = (lx + rx) / 2.0
    eye_cy = (top_y + bot_y) / 2.0
    return (cx - eye_cx) / (eye_w + 1e-6), (cy - eye_cy) / (eye_h + 1e-6)


# =============================================================
# Helper: draw a calibration/drift dot with progress ring
# =============================================================
def draw_target_dot(canvas, tx, ty, progress, label, total_label):
    angle = int(360 * progress)
    cv2.ellipse(canvas, (tx, ty), (22, 22), -90, 0, angle, (0, 255, 100), 3)
    cv2.circle(canvas, (tx, ty), 12, (0, 200, 80), -1)
    text = f"{label}  {total_label}"
    cv2.putText(canvas, text,
                ((SCREEN_WIDTH - 420) // 2, SCREEN_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)


# =============================================================
# Main Application
# =============================================================
cv2.namedWindow('Sandbox (Your Screen)', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Sandbox (Your Screen)', cv2.WND_PROP_FULLSCREEN,
                       cv2.WINDOW_FULLSCREEN)
cv2.namedWindow('Camera Feed')

print("Starting camera thread…")
vs = WebcamVideoStream(src=0, width=1280, height=720).start()
time.sleep(1.0)
print("Webcam ready.")
print("  SPACE  → confirm calibration dot")
print("  d      → start drift correction (5 quick dots, no full recalibration)")
print("  r      → full recalibration")
print("  q      → quit")

kalman = KalmanGaze()

lrx = lry = rrx = rry = 0.0
head_yaw = head_pitch = 0.0
face_scale_val = 0.05
current_feat   = None   # most recent 11-D feature (used by drift capture)

smooth_x, smooth_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
current_landmarks   = None
last_frame_id       = -1
display_image       = None

while True:
    current_frame_id = vs.frame_id
    sandbox = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) * 30

    # ----------------------------------------------------------------
    # 1. Send new frame to MediaPipe
    # ----------------------------------------------------------------
    if current_frame_id > last_frame_id:
        raw_frame = vs.read()
        if raw_frame is None:
            continue
        image         = cv2.flip(raw_frame, 1)
        display_image = image.copy()
        ts_ms         = int(time.time() * 1000)
        mp_image      = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        detector.detect_async(mp_image, ts_ms)
        last_frame_id = current_frame_id
    else:
        if display_image is None:
            continue
        image = display_image.copy()

    h, w = image.shape[:2]

    # ----------------------------------------------------------------
    # 2. Consume latest ML result
    # ----------------------------------------------------------------
    process_new_frame = False
    with landmark_lock:
        if new_data_available:
            current_landmarks  = latest_landmarks
            new_data_available = False
            process_new_frame  = True

    # ----------------------------------------------------------------
    # 3. Compute features on fresh, open-eyed frames
    # ----------------------------------------------------------------
    face_found = False
    if process_new_frame and current_landmarks:
        face_found = True
        lm_px = [(lm.x * w, lm.y * h) for lm in current_landmarks]

        yaw, pitch, _ = get_head_pose(current_landmarks, w, h)
        fs             = get_face_scale(current_landmarks)

        left_x,  left_y  = get_eye_gaze_ratio(
            lm_px, IRIS_LEFT,  EYE_LEFT_OUTER,  EYE_LEFT_INNER,
            EYE_LEFT_TOP,  EYE_LEFT_BOTTOM)
        right_x, right_y = get_eye_gaze_ratio(
            lm_px, IRIS_RIGHT, EYE_RIGHT_OUTER, EYE_RIGHT_INNER,
            EYE_RIGHT_TOP, EYE_RIGHT_BOTTOM)

        eyes_open = (left_x is not None) and (right_x is not None)

        if eyes_open:
            lrx, lry = left_x,  left_y
            rrx, rry = right_x, right_y
            head_yaw, head_pitch = yaw, pitch
            face_scale_val = fs

            current_feat = build_feature(lrx, lry, rrx, rry,
                                         head_yaw, head_pitch, face_scale_val)

            if calib_done:
                sx, sy = apply_calibration(current_feat)
                kx, ky = kalman.update(int(sx * SCREEN_WIDTH),
                                       int(sy * SCREEN_HEIGHT))
                smooth_x, smooth_y = int(kx), int(ky)

            # Fill sample buffers (calibration & drift use the same gate)
            if sampling_active:
                sample_buffer.append(current_feat.tolist())
            if drift_sampling:
                drift_sample_buf.append(current_feat.tolist())

    # ----------------------------------------------------------------
    # Draw iris dots
    # ----------------------------------------------------------------
    if current_landmarks:
        lm_draw = [(lm.x * w, lm.y * h) for lm in current_landmarks]
        for idx_group in [IRIS_LEFT, IRIS_RIGHT]:
            pts  = [lm_draw[i] for i in idx_group]
            icx  = int(sum(p[0] for p in pts) / len(pts))
            icy  = int(sum(p[1] for p in pts) / len(pts))
            cv2.circle(image, (icx, icy), 4, (0, 215, 255), -1)

    # ================================================================
    # UI BRANCH A: Full calibration
    # ================================================================
    if not calib_done:
        tx = int(CALIB_POINTS_NORM[calib_index][0] * SCREEN_WIDTH)
        ty = int(CALIB_POINTS_NORM[calib_index][1] * SCREEN_HEIGHT)

        if sampling_active:
            draw_target_dot(sandbox, tx, ty,
                            len(sample_buffer) / SAMPLES_NEEDED,
                            "Hold still…",
                            f"{len(sample_buffer)}/{SAMPLES_NEEDED}")

            if len(sample_buffer) >= SAMPLES_NEEDED:
                clean      = reject_outliers(sample_buffer, k=2.0)
                avg        = clean.mean(axis=0)
                variance   = clean.var(axis=0).mean()
                confidence = 1.0 / (variance + 1e-6)

                calib_features.append(avg)
                calib_targets.append(list(CALIB_POINTS_NORM[calib_index]))
                calib_weights.append(confidence)

                print(f"Point {calib_index+1} captured "
                      f"({len(clean)}/{len(sample_buffer)} kept)  "
                      f"yaw={avg[4]*30:.1f}°  pitch={avg[5]*20:.1f}°")

                sample_buffer.clear()
                sampling_active = False
                calib_index    += 1

                if calib_index == len(CALIB_POINTS_NORM):
                    fit_calibration(calib_features, calib_targets, calib_weights)
                    calib_done = True
                    kalman.reset()
                    print("Calibration complete! Tracking active.")
                    print("Press 'd' to run a quick drift correction at any time.")
        else:
            pulse = int(10 + 6 * abs(np.sin(time.time() * 3)))
            cv2.circle(sandbox, (tx, ty), pulse + 6, (255, 255, 255), 2)
            cv2.circle(sandbox, (tx, ty), pulse,     (0, 0, 220),     -1)
            text  = f"Look at dot ({calib_index+1}/{len(CALIB_POINTS_NORM)}) — press SPACE"
            tsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            cv2.putText(sandbox, text,
                        ((SCREEN_WIDTH - tsize[0]) // 2, SCREEN_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)

        # Draw all dot indicators
        for i, (px, py) in enumerate(CALIB_POINTS_NORM):
            if   i < calib_index:   color, size = (0, 200, 0),    8
            elif i == calib_index:  color, size = (255, 255, 255), 5
            else:                   color, size = (80, 80, 80),    6
            cv2.circle(sandbox,
                       (int(px * SCREEN_WIDTH), int(py * SCREEN_HEIGHT)),
                       size, color, -1)

    # ================================================================
    # UI BRANCH B: Tracking (+ optional drift correction overlay)
    # ================================================================
    else:
        # ------ Drift correction session --------------------------------
        if drift_mode:
            dx = int(DRIFT_POINTS_NORM[drift_index][0] * SCREEN_WIDTH)
            dy = int(DRIFT_POINTS_NORM[drift_index][1] * SCREEN_HEIGHT)

            # Small hint so user knows what's happening
            hint = (f"Drift correction  ({drift_index+1}/{len(DRIFT_POINTS_NORM)}) "
                    f"— look at dot, press SPACE")
            cv2.putText(sandbox, hint,
                        (20, SCREEN_HEIGHT - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 220, 50), 2)

            if drift_sampling:
                draw_target_dot(sandbox, dx, dy,
                                len(drift_sample_buf) / DRIFT_SAMPLES_NEED,
                                "Hold still…",
                                f"{len(drift_sample_buf)}/{DRIFT_SAMPLES_NEED}")

                if len(drift_sample_buf) >= DRIFT_SAMPLES_NEED:
                    # Reject outliers, average, record correction
                    clean = reject_outliers(drift_sample_buf, k=2.0)
                    avg   = clean.mean(axis=0)
                    true_x, true_y = DRIFT_POINTS_NORM[drift_index]
                    add_drift_correction(avg, true_x, true_y)

                    drift_sample_buf.clear()
                    drift_sampling = False
                    drift_index   += 1

                    if drift_index >= len(DRIFT_POINTS_NORM):
                        # Done — exit drift mode
                        drift_mode  = False
                        drift_index = 0
                        kalman.reset()   # reset Kalman so it re-locks quickly
                        print("Drift correction complete.")
            else:
                # Waiting for SPACE
                pulse = int(8 + 5 * abs(np.sin(time.time() * 3)))
                cv2.circle(sandbox, (dx, dy), pulse + 5, (255, 220,  50), 2)
                cv2.circle(sandbox, (dx, dy), pulse,     (200, 160,   0), -1)

        # ------ Normal gaze dot -----------------------------------------
        else:
            cv2.circle(sandbox, (smooth_x, smooth_y), 30, (0,   0, 100), -1)
            cv2.circle(sandbox, (smooth_x, smooth_y), 18, (0,   0, 255), -1)
            cv2.circle(sandbox, (smooth_x, smooth_y),  6, (255, 255, 255), -1)

            cv2.putText(sandbox,
                        "Press 'd' to drift-correct  |  'r' to recalibrate  |  'q' to quit",
                        (20, SCREEN_HEIGHT - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 1)

        # Camera feed HUD (always shown during tracking)
        cv2.putText(image, "Tracking Active — 'r' recalibrate  'd' drift fix",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image,
                    f"L({lrx:.3f},{lry:.3f})  R({rrx:.3f},{rry:.3f})"
                    f"  yaw={head_yaw:.1f}°  pitch={head_pitch:.1f}°"
                    f"  drift_fixes={len(drift_corrections)}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow('Sandbox (Your Screen)', sandbox)
    cv2.imshow('Camera Feed', image)

    key = cv2.waitKey(1) & 0xFF

    # ----------------------------------------------------------------
    # Key handling
    # ----------------------------------------------------------------
    if key == ord('q'):
        break

    elif key == ord('r'):
        # Full recalibration — reset everything
        calib_index    = 0
        calib_features.clear()
        calib_targets.clear()
        calib_weights.clear()
        calib_done     = False
        sampling_active = False
        sample_buffer.clear()
        drift_mode     = False
        drift_index    = 0
        drift_sampling = False
        drift_sample_buf.clear()
        drift_corrections.clear()
        smooth_x, smooth_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        kalman.reset()
        cv2.setWindowProperty('Sandbox (Your Screen)',
                               cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        print("Recalibrating…")

    elif key == ord('d') and calib_done and not drift_mode:
        # Start a quick drift-correction session
        drift_mode     = True
        drift_index    = 0
        drift_sampling = False
        drift_sample_buf.clear()
        print("Drift correction started — look at each yellow dot and press SPACE.")

    elif key == ord(' '):
        if not calib_done and not sampling_active:
            # Calibration: start sampling current dot
            sampling_active = True
            sample_buffer.clear()
            print(f"Sampling calibration point {calib_index+1}… hold still!")

        elif calib_done and drift_mode and not drift_sampling:
            # Drift: start sampling current drift dot
            drift_sampling = True
            drift_sample_buf.clear()
            print(f"Sampling drift point {drift_index+1}… hold still!")

vs.stop()
detector.close()
cv2.destroyAllWindows()