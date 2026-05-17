"""
Robust Eye Gaze Tracker
=======================
Improvements in this version:
  1–7. All previous improvements (head pose, poly features, Kalman, drift, etc.)
  8.   Output range stretching  → fixes slow/unreachable corners
  9.   Gaussian heatmap blob    → replaces single dot with decaying heat splat
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
    def __init__(self, process_noise=1e-4, measurement_noise=8e-2, dt=1/30):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array(
            [[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self._initialized = False

    def update(self, x, y):
        if not self._initialized:
            self.kf.statePre  = np.array([[x],[y],[0],[0]], dtype=np.float32)
            self.kf.statePost = self.kf.statePre.copy()
            self._initialized = True
            return x, y
        self.kf.predict()
        corrected = self.kf.correct(np.array([[x],[y]], dtype=np.float32))
        return float(corrected[0]), float(corrected[1])

    def reset(self):
        self._initialized = False


# =============================================================
# Feature Engineering
# =============================================================
def build_feature(lx, ly, rx, ry, yaw, pitch, face_scale):
    yn = yaw   / 30.0
    pn = pitch / 20.0
    return np.array([
        lx, ly, rx, ry,
        yn, pn,
        face_scale,
        lx * yn, rx * yn,
        ly * pn, ry * pn,
    ], dtype=np.float64)
def get_iris_features(lm_px):
    """
    Iris offset from nose tip, normalized by inter-ocular distance.
    Both iris and nose are on the rigid face so they move together on
    head rotation — cancelling the foreshortening that caused drift.
    """
    nose   = np.array(lm_px[1], dtype=np.float64)
    l_iris = np.mean([np.array(lm_px[i], dtype=np.float64) for i in IRIS_LEFT],  axis=0)
    r_iris = np.mean([np.array(lm_px[i], dtype=np.float64) for i in IRIS_RIGHT], axis=0)

    iod = np.hypot(lm_px[263][0] - lm_px[33][0],
                   lm_px[263][1] - lm_px[33][1]) + 1e-6

    return ((l_iris[0] - nose[0]) / iod,
            (l_iris[1] - nose[1]) / iod,
            (r_iris[0] - nose[0]) / iod,
            (r_iris[1] - nose[1]) / iod)

# =============================================================
# ML Models
# =============================================================
from sklearn.ensemble import GradientBoostingRegressor

def _make_model():
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                  learning_rate=0.1, random_state=0),
    )

model_x = _make_model()
model_y = _make_model()

def fit_calibration(features, targets, weights=None):
    X = np.array(features)
    Y = np.array(targets)
    sw = None
    if weights is not None and len(weights) == len(X):
        W = np.clip(np.array(weights, dtype=np.float64), 1e-6, None)
        sw = W / W.max()
    model_x.fit(X, Y[:, 0], gradientboostingregressor__sample_weight=sw)
    model_y.fit(X, Y[:, 1], gradientboostingregressor__sample_weight=sw)


def apply_calibration(feat):
    pred = feat.reshape(1, -1)
    return float(model_x.predict(pred)[0]), float(model_y.predict(pred)[0])


# =============================================================
# Output Range Stretcher  (NEW — fixes slow/unreachable corners)
# =============================================================
class RangeStretcher:
    """
    Problem: SVR with RBF kernel shrinks predictions toward the mean.
    Even if a corner is (0.05, 0.05) in the training data, the model
    may only predict (0.12, 0.14) — so looking at the corner moves the
    dot slowly toward 0.12 but never reaches 0.05.

    Solution: after training, run the model on all calibration inputs,
    observe the actual min/max of its output, then fit a linear map:

        stretched = (raw_pred - pred_min) / (pred_max - pred_min)

    so the full [0, 1] screen range is reachable again.

    margin: shrinks the observed extremes slightly inward before
    computing the stretch, making the map a little more aggressive
    so edge predictions genuinely reach the corners.
    """
    def __init__(self, margin=0.02):
        self.margin  = margin
        self.x_min = self.x_max = None
        self.y_min = self.y_max = None
        self.fitted  = False

    def fit(self, features, targets):
        X     = np.array(features)
        raw_x = model_x.predict(X)
        raw_y = model_y.predict(X)

        self.x_min, self.x_max = raw_x.min(), raw_x.max()
        self.y_min, self.y_max = raw_y.min(), raw_y.max()

        # Shrink observed range inward so stretch is slightly aggressive
        span_x = self.x_max - self.x_min
        span_y = self.y_max - self.y_min
        self.x_min += span_x * self.margin
        self.x_max -= span_x * self.margin
        self.y_min += span_y * self.margin
        self.y_max -= span_y * self.margin

        self.fitted = True
        print(f"[Stretcher] X=[{self.x_min:.3f}, {self.x_max:.3f}]  "
              f"Y=[{self.y_min:.3f}, {self.y_max:.3f}]")

    def apply(self, raw_x, raw_y):
        if not self.fitted:
            return np.clip(raw_x, 0, 1), np.clip(raw_y, 0, 1)
        sx = (raw_x - self.x_min) / max(self.x_max - self.x_min, 1e-6)
        sy = (raw_y - self.y_min) / max(self.y_max - self.y_min, 1e-6)
        return float(np.clip(sx, 0, 1)), float(np.clip(sy, 0, 1))


stretcher = RangeStretcher(margin=0.02)


def apply_calibration_stretched(feat):
    raw_x, raw_y = apply_calibration(feat)
    return stretcher.apply(raw_x, raw_y)


# =============================================================
# Gaussian Heatmap Blob  (NEW)
# =============================================================
class GazeHeatmap:
    """
    How professional eye-trackers draw gaze:

    Instead of a single moving dot, they maintain a float32 accumulator
    (same resolution as the display).  Every frame:
      1. Multiply accumulator by decay (<1) — old gaze fades out
      2. Add a Gaussian "splat" centred at the current gaze point
      3. Normalise, colourise with a heatmap LUT, alpha-blend onto canvas

    Result: fixations produce a bright tight blob; saccades leave a faint
    smeared trail.  The blob size (sigma) represents spatial uncertainty.

    Parameters
    ----------
    blob_sigma  pixels of the 1-SD radius.  55px ≈ 3 cm on a 27" 1080p
                screen viewed from 60 cm — a realistic fixation spread.
    decay       fraction kept each frame.  0.93 @ 30fps ≈ half-life ~9 frames
    alpha       heatmap opacity over the dark sandbox background
    """
    def __init__(self, width=1920, height=1080,
                 blob_sigma=55.0, decay=0.93, alpha=0.55):
        self.w, self.h = width, height
        self.sigma     = blob_sigma
        self.decay     = decay
        self.alpha     = alpha
        self.acc       = np.zeros((height, width), dtype=np.float32)

        # Pre-compute Gaussian kernel patch (avoids per-frame computation)
        r       = int(blob_sigma * 3.5)
        ksize   = 2 * r + 1
        ax      = np.arange(-r, r + 1, dtype=np.float32)
        xx, yy  = np.meshgrid(ax, ax)
        kernel  = np.exp(-(xx**2 + yy**2) / (2.0 * blob_sigma**2))
        self._kernel = (kernel / kernel.max()).astype(np.float32)
        self._r      = r

    def update(self, gx: int, gy: int):
        """Decay accumulator and add a new Gaussian splat at (gx, gy)."""
        self.acc *= self.decay

        r  = self._r
        x0 = max(gx - r, 0);       x1 = min(gx + r + 1, self.w)
        y0 = max(gy - r, 0);       y1 = min(gy + r + 1, self.h)
        kx0 = x0 - (gx - r);      ky0 = y0 - (gy - r)
        kx1 = kx0 + (x1 - x0);    ky1 = ky0 + (y1 - y0)
        self.acc[y0:y1, x0:x1] += self._kernel[ky0:ky1, kx0:kx1]

    def render(self, canvas: np.ndarray) -> np.ndarray:
        """Blend heatmap onto canvas in-place and return it."""
        peak = self.acc.max()
        if peak < 1e-3:
            return canvas

        norm  = np.clip(self.acc / peak, 0.0, 1.0)
        u8    = (norm * 255).astype(np.uint8)
        color = cv2.applyColorMap(u8, cv2.COLORMAP_JET)

        # Pixels below threshold stay transparent (keeps background clean)
        mask      = (norm > 0.05).astype(np.float32)
        alpha_map = (norm * self.alpha * mask)

        for c in range(3):
            canvas[:, :, c] = np.clip(
                color[:, :, c] * alpha_map
                + canvas[:, :, c] * (1.0 - alpha_map),
                0, 255).astype(np.uint8)
        return canvas

    def reset(self):
        self.acc[:] = 0.0


SCREEN_WIDTH  = 1920
SCREEN_HEIGHT = 1080

heatmap = GazeHeatmap(
    width=SCREEN_WIDTH, height=SCREEN_HEIGHT,
    blob_sigma=55.0, decay=0.93, alpha=0.55,
)


# =============================================================
# Head Pose
# =============================================================
_HEAD_3D = np.array([
    [  0.0,    0.0,    0.0],
    [  0.0, -330.0,  -65.0],
    [-225.0,  170.0, -135.0],
    [ 225.0,  170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [ 150.0, -150.0, -125.0],
], dtype=np.float64)
_HEAD_IDX = [1, 152, 33, 263, 61, 291]

def get_head_pose(lm_norm, img_w, img_h):
    pts2d = np.array(
        [(lm_norm[i].x * img_w, lm_norm[i].y * img_h) for i in _HEAD_IDX],
        dtype=np.float64)
    fl      = float(img_w)
    cam_mat = np.array([[fl,0,img_w/2],[0,fl,img_h/2],[0,0,1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        _HEAD_3D, pts2d, cam_mat, np.zeros((4,1)),
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0, np.eye(3)          # <-- add R
    R, _ = cv2.Rodrigues(rvec)
    sy    = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        yaw   = np.degrees(np.arctan2(-R[2,0], sy))     # FIXED: Y-axis rotation (Left/Right)
        pitch = np.degrees(np.arctan2( R[2,1], R[2,2]))
        roll  = np.degrees(np.arctan2( R[1,0], R[0,0]))
    else:
        pitch = np.degrees(np.arctan2(-R[2,0], sy))
        yaw, roll = 0.0, np.degrees(np.arctan2(-R[0,1], R[1,1]))
    return yaw, pitch, roll, R                    # <-- add R

def get_face_scale(lm_norm):
    l, r = lm_norm[33], lm_norm[263]
    return float(np.hypot(r.x - l.x, r.y - l.y))

def get_compensated_gaze(lm_px, R, img_w, img_h):
    """
    Iris position relative to nose tip, rotated into a head-pose-normalized
    frame using the inverse of the head rotation matrix.
    This removes the perspective artifact that causes opposite-direction drift.
    """
    nose = np.array(lm_px[1], dtype=np.float64)

    def iris_center(indices):
        pts = [np.array(lm_px[i], dtype=np.float64) for i in indices]
        return np.mean(pts, axis=0)

    lc = iris_center(IRIS_LEFT)
    rc = iris_center(IRIS_RIGHT)

    # Face width in pixels for normalization
    face_w = np.hypot(lm_px[263][0] - lm_px[33][0],
                      lm_px[263][1] - lm_px[33][1]) + 1e-6

    # Offset from nose, normalized — z=0 (we work in 2D image plane)
    def compensate(pt):
        v = np.array([(pt[0] - nose[0]) / face_w,
                      (pt[1] - nose[1]) / face_w,
                      0.0])
        # Rotate by R^T to undo head rotation
        vc = R.T @ v
        return float(vc[0]), float(vc[1])

    lx, ly = compensate(lc)
    rx, ry = compensate(rc)
    return lx, ly, rx, ry
# =============================================================
# Outlier Rejection
# =============================================================
def reject_outliers(samples, k=2.0):
    arr    = np.array(samples)
    median = np.median(arr, axis=0)
    mad    = np.median(np.abs(arr - median), axis=0) + 1e-9
    mask   = np.all(np.abs(arr - median) <= k * mad, axis=1)
    good   = arr[mask]
    return good if len(good) >= max(5, len(arr)//4) else arr


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

    def read(self):  return self.frame
    def stop(self):
        self.stopped = True
        self.stream.release()


# =============================================================
# Constants
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

DRIFT_POINTS_NORM = [
    (0.50, 0.50),
    (0.05, 0.05), (0.95, 0.05),
    (0.05, 0.95), (0.95, 0.95),
]

SAMPLES_NEEDED     = 15
DRIFT_SAMPLES_NEED = 10

# =============================================================
# Calibration state
# =============================================================
calib_index    = 0
calib_features = []
calib_targets  = []
calib_weights  = []
calib_done     = False

sampling_active = False
sample_buffer   = []

stale_feat_frames = 0
MAX_STALE_FRAMES  = 15
# =============================================================
# Drift state
# =============================================================
drift_corrections = []


def add_drift_correction(feat, true_x_norm, true_y_norm):
    drift_corrections.append((feat.copy(), [true_x_norm, true_y_norm]))
    if len(drift_corrections) > 30:
        drift_corrections.pop(0)
    all_feats   = calib_features + [d[0] for d in drift_corrections]
    all_targets = calib_targets  + [d[1] for d in drift_corrections]
    base_w      = max(calib_weights, default=1.0)
    all_weights = calib_weights + [3.0 * base_w] * len(drift_corrections)
    fit_calibration(all_feats, all_targets, all_weights)
    stretcher.fit(all_feats, all_targets)
    print(f"[Drift] Retrained — {len(drift_corrections)} drift fix(es).")


drift_mode       = False
drift_index      = 0
drift_sampling   = False
drift_sample_buf = []

# =============================================================
# MediaPipe
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
    ear    = eye_h / (eye_w + 1e-6)
    if ear < 0.10:
        return None, None
    eye_cx = (lx + rx) / 2.0
    eye_cy = (top_y + bot_y) / 2.0
    return (cx - eye_cx) / (eye_w + 1e-6), (cy - eye_cy) / (eye_h + 1e-6)


# =============================================================
# UI Helper
# =============================================================
def draw_target_dot(canvas, tx, ty, progress, label):
    angle = int(360 * progress)
    cv2.ellipse(canvas, (tx, ty), (22, 22), -90, 0, angle, (0, 255, 100), 3)
    cv2.circle(canvas,  (tx, ty), 12, (0, 200, 80), -1)
    cv2.putText(canvas, label,
                ((SCREEN_WIDTH - 440) // 2, SCREEN_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)


# =============================================================
# Main loop
# =============================================================
cv2.namedWindow('Sandbox (Your Screen)', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Sandbox (Your Screen)',
                       cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.namedWindow('Camera Feed')

print("Starting camera…")
vs = WebcamVideoStream(src=0, width=1280, height=720).start()
time.sleep(1.0)
print("Ready.  SPACE=confirm dot | d=drift fix | r=recalibrate | q=quit")

kalman = KalmanGaze()

lrx = lry = rrx = rry = 0.0
head_yaw = head_pitch  = 0.0
face_scale_val = 0.05
current_feat   = None

smooth_x, smooth_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
current_landmarks   = None
last_frame_id       = -1
display_image       = None
training_in_progress = False
while True:
    current_frame_id = vs.frame_id
    sandbox = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) * 18

    # ------------------------------------------------------------------
    # 1. Feed frame to MediaPipe
    # ------------------------------------------------------------------
    if current_frame_id > last_frame_id:
        raw_frame = vs.read()
        if raw_frame is None:
            continue
        image         = cv2.flip(raw_frame, 1)
        display_image = image.copy()
        ts_ms         = int(time.time() * 1000)
        mp_image      = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))
        detector.detect_async(mp_image, ts_ms)
        last_frame_id = current_frame_id
    else:
        if display_image is None:
            continue
        image = display_image.copy()

    h, w = image.shape[:2]

    # ------------------------------------------------------------------
    # 2. Consume ML result
    # ------------------------------------------------------------------
    process_new_frame = False
    with landmark_lock:
        if new_data_available:
            current_landmarks  = latest_landmarks
            new_data_available = False
            process_new_frame  = True

    # ------------------------------------------------------------------
    # 3. Feature extraction & prediction
    # ------------------------------------------------------------------
    face_found = False
    if process_new_frame and current_landmarks:
        face_found = True
        lm_px = [(lm.x * w, lm.y * h) for lm in current_landmarks]

        yaw, pitch, roll, R = get_head_pose(current_landmarks, w, h)
        fs             = get_face_scale(current_landmarks)

        left_x,  left_y  = get_eye_gaze_ratio(
            lm_px, IRIS_LEFT,  EYE_LEFT_OUTER,  EYE_LEFT_INNER,
            EYE_LEFT_TOP,  EYE_LEFT_BOTTOM)
        right_x, right_y = get_eye_gaze_ratio(
            lm_px, IRIS_RIGHT, EYE_RIGHT_OUTER, EYE_RIGHT_INNER,
            EYE_RIGHT_TOP, EYE_RIGHT_BOTTOM)

                # In the feature extraction block, change:
        eyes_open = (left_x is not None) and (right_x is not None)

        use_feat = eyes_open or (current_feat is not None and stale_feat_frames < MAX_STALE_FRAMES)

        if use_feat:
            if eyes_open:
                lrx, lry, rrx, rry = get_compensated_gaze(lm_px, R, w, h)

                head_yaw, head_pitch = yaw, pitch
                face_scale_val = fs
                current_feat = build_feature(lrx, lry, rrx, rry,
                                            head_yaw, head_pitch, face_scale_val)
                stale_feat_frames = 0
            else:
                stale_feat_frames += 1

            if calib_done and current_feat is not None:
                sx, sy   = apply_calibration_stretched(current_feat)
                raw_px   = int(sx * SCREEN_WIDTH)
                raw_py   = int(sy * SCREEN_HEIGHT)
                kx, ky   = kalman.update(raw_px, raw_py)
                smooth_x = int(np.clip(kx, 0, SCREEN_WIDTH  - 1))
                smooth_y = int(np.clip(ky, 0, SCREEN_HEIGHT - 1))
                heatmap.update(smooth_x, smooth_y)

            if sampling_active and eyes_open:
                sample_buffer.append(current_feat.tolist())
            if drift_sampling and eyes_open:
                drift_sample_buf.append(current_feat.tolist())
        # ------------------------------------------------------------------
    # Draw iris dots on camera feed
    # ------------------------------------------------------------------
    if current_landmarks:
        lm_draw = [(lm.x * w, lm.y * h) for lm in current_landmarks]
        for idx_group in [IRIS_LEFT, IRIS_RIGHT]:
            pts  = [lm_draw[i] for i in idx_group]
            icx  = int(sum(p[0] for p in pts) / len(pts))
            icy  = int(sum(p[1] for p in pts) / len(pts))
            cv2.circle(image, (icx, icy), 4, (0, 215, 255), -1)

    # ==================================================================
    # CALIBRATION UI
    # ==================================================================
    if not calib_done:
        if training_in_progress:
            cv2.putText(sandbox, "Training model, please wait…",
                        (SCREEN_WIDTH//2 - 280, SCREEN_HEIGHT//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 200, 255), 2)
            cv2.imshow('Sandbox (Your Screen)', sandbox)
            cv2.imshow('Camera Feed', image)
            cv2.waitKey(1)
            continue

        tx = int(CALIB_POINTS_NORM[calib_index][0] * SCREEN_WIDTH)
        ty = int(CALIB_POINTS_NORM[calib_index][1] * SCREEN_HEIGHT)

        if sampling_active:
            draw_target_dot(sandbox, tx, ty,
                            len(sample_buffer) / SAMPLES_NEEDED,
                            f"Hold still…  {len(sample_buffer)}/{SAMPLES_NEEDED}")

            if len(sample_buffer) >= SAMPLES_NEEDED:
                clean      = reject_outliers(sample_buffer, k=2.0)
                avg        = clean.mean(axis=0)
                variance   = clean.var(axis=0).mean()
                confidence = 1.0 / (variance + 1e-6)

                calib_features.append(avg)
                calib_targets.append(list(CALIB_POINTS_NORM[calib_index]))
                calib_weights.append(confidence)

                print(f"Point {calib_index+1} captured "
                      f"({len(clean)}/{len(sample_buffer)} kept)")

                sample_buffer.clear()
                sampling_active = False
                calib_index    += 1

                            # With this:
                if calib_index == len(CALIB_POINTS_NORM):
                    calib_done = False  # stays False until thread finishes
                    training_in_progress = True

                    def _train():
                        global calib_done, training_in_progress
                        fit_calibration(calib_features, calib_targets, calib_weights)
                        stretcher.fit(calib_features, calib_targets)
                        kalman.reset()
                        heatmap.reset()
                        calib_done = True
                        training_in_progress = False
                        print("Calibration complete!  Press 'd' for drift fix.")

                    threading.Thread(target=_train, daemon=True).start()
        
            
        else:
            pulse = int(10 + 6 * abs(np.sin(time.time() * 3)))
            cv2.circle(sandbox, (tx, ty), pulse + 6, (255, 255, 255), 2)
            cv2.circle(sandbox, (tx, ty), pulse,     (0, 0, 220),     -1)
            text  = f"Look at dot ({calib_index+1}/{len(CALIB_POINTS_NORM)}) — press SPACE"
            tsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            cv2.putText(sandbox, text,
                        ((SCREEN_WIDTH - tsize[0]) // 2, SCREEN_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)

        for i, (px, py) in enumerate(CALIB_POINTS_NORM):
            if   i < calib_index:   color, size = (0, 200, 0),    8
            elif i == calib_index:  color, size = (255, 255, 255), 5
            else:                   color, size = (80, 80, 80),    6
            cv2.circle(sandbox,
                       (int(px * SCREEN_WIDTH), int(py * SCREEN_HEIGHT)),
                       size, color, -1)

    # ==================================================================
    # TRACKING UI
    # ==================================================================
    else:
        # Layer 1: decaying heatmap blob
        heatmap.render(sandbox)

        # Layer 2: small white crosshair pinpoints exact predicted position
        cx, cy = smooth_x, smooth_y
        cv2.line(sandbox, (cx - 14, cy), (cx + 14, cy), (255, 255, 255), 1)
        cv2.line(sandbox, (cx, cy - 14), (cx, cy + 14), (255, 255, 255), 1)
        cv2.circle(sandbox, (cx, cy), 4, (255, 255, 255), -1)

        # Drift correction overlay
        if drift_mode:
            dx = int(DRIFT_POINTS_NORM[drift_index][0] * SCREEN_WIDTH)
            dy = int(DRIFT_POINTS_NORM[drift_index][1] * SCREEN_HEIGHT)
            hint = (f"Drift correction ({drift_index+1}/{len(DRIFT_POINTS_NORM)}) "
                    f"— look at dot, press SPACE")
            cv2.putText(sandbox, hint, (20, SCREEN_HEIGHT - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 220, 50), 2)

            if drift_sampling:
                draw_target_dot(sandbox, dx, dy,
                                len(drift_sample_buf) / DRIFT_SAMPLES_NEED,
                                f"Hold still… {len(drift_sample_buf)}/{DRIFT_SAMPLES_NEED}")

                if len(drift_sample_buf) >= DRIFT_SAMPLES_NEED:
                    clean  = reject_outliers(drift_sample_buf, k=2.0)
                    avg    = clean.mean(axis=0)
                    tx_n, ty_n = DRIFT_POINTS_NORM[drift_index]
                    add_drift_correction(avg, tx_n, ty_n)
                    drift_sample_buf.clear()
                    drift_sampling = False
                    drift_index   += 1
                    if drift_index >= len(DRIFT_POINTS_NORM):
                        drift_mode  = False
                        drift_index = 0
                        kalman.reset()
                        heatmap.reset()
                        print("Drift correction complete.")
            else:
                pulse = int(8 + 5 * abs(np.sin(time.time() * 3)))
                cv2.circle(sandbox, (dx, dy), pulse + 5, (255, 220, 50), 2)
                cv2.circle(sandbox, (dx, dy), pulse,     (200, 160,  0), -1)
        else:
            cv2.putText(sandbox,
                        "d = drift fix   r = recalibrate   q = quit",
                        (20, SCREEN_HEIGHT - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 100, 100), 1)

        cv2.putText(image, "Tracking — 'd' drift  'r' recalibrate",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image,
                    f"L({lrx:.3f},{lry:.3f})  R({rrx:.3f},{rry:.3f})"
                    f"  yaw={head_yaw:.1f}  pitch={head_pitch:.1f}"
                    f"  fixes={len(drift_corrections)}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200,200,200), 1)

    cv2.imshow('Sandbox (Your Screen)', sandbox)
    cv2.imshow('Camera Feed', image)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('r'):
        calib_index = 0
        calib_features.clear(); calib_targets.clear(); calib_weights.clear()
        calib_done      = False
        sampling_active = False; sample_buffer.clear()
        drift_mode      = False; drift_index = 0
        drift_sampling  = False; drift_sample_buf.clear()
        drift_corrections.clear()
        smooth_x, smooth_y = SCREEN_WIDTH//2, SCREEN_HEIGHT//2
        kalman.reset(); heatmap.reset()
        cv2.setWindowProperty('Sandbox (Your Screen)',
                               cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        print("Recalibrating…")

    elif key == ord('d') and calib_done and not drift_mode:
        drift_mode = True; drift_index = 0
        drift_sampling = False; drift_sample_buf.clear()
        print("Drift correction — look at each yellow dot and press SPACE.")

    elif key == ord(' '):
        if not calib_done and not sampling_active:
            sampling_active = True
            sample_buffer.clear()
            print(f"Sampling point {calib_index+1}…")
        elif calib_done and drift_mode and not drift_sampling:
            drift_sampling = True
            drift_sample_buf.clear()
            print(f"Sampling drift point {drift_index+1}…")

vs.stop()
detector.close()
cv2.destroyAllWindows()