"""
gaze_core.py — Reusable gaze estimation module
===============================================
RetinaFace + ResNet50 eye crops + PCA/Ridge calibration, wrapped in a
single GazeReader class your teammates can import and call.
"""

import cv2, time, numpy as np, threading, sys
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

try:
    from face_detection import RetinaFace
except ImportError:
    print("[ERROR] pip install git+https://github.com/Ahmednull/L2CS-Net.git")
    sys.exit(1)

# Default calibration grid: 4 corner anchors + 3×3 dense grid in reading column
CALIB_PTS = [
    (0.15, 0.15), (0.85, 0.15),
    (0.15, 0.85), (0.85, 0.85),
    (0.35, 0.15), (0.50, 0.15), (0.65, 0.15),
    (0.35, 0.50), (0.50, 0.50), (0.65, 0.50),
    (0.35, 0.85), (0.50, 0.85), (0.65, 0.85),
]
DRIFT_PTS = [
    (0.50, 0.50),
    (0.35, 0.15), (0.65, 0.15),
    (0.35, 0.85), (0.65, 0.85),
]


# ── Kalman filter ─────────────────────────────────────────────────────────────
class KalmanGaze:
    def __init__(self, pn=1e-3, mn=3e-2, dt=1/30):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix    = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        self.kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * pn
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * mn
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self._ok = False

    def update(self, x, y):
        if not self._ok:
            self.kf.statePre = self.kf.statePost = np.array([[x],[y],[0],[0]], dtype=np.float32)
            self._ok = True; return x, y
        self.kf.predict()
        c = self.kf.correct(np.array([[x],[y]], dtype=np.float32))
        return float(c[0,0]), float(c[1,0])

    def reset(self): self._ok = False


# ── Background CNN inference ───────────────────────────────────────────────────
class _GazeEstimator:
    """
    Internal: runs RetinaFace + ResNet50 eye crops in a background thread.
    Submit frames via submit(); read results via read().
    """
    _transform = T.Compose([
        T.ToPILImage(),
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, device):
        gpu_id = -1 if device.type == 'cpu' else (device.index or 0)
        self._detector = RetinaFace(gpu_id=gpu_id)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self._extractor = nn.Sequential(*list(resnet.children())[:-1])
        self._extractor.eval()
        for p in self._extractor.parameters():
            p.requires_grad_(False)
        self._extractor.to(device)
        self._device = device
        self._lock    = threading.Lock()
        self._pending = None
        self.features = None
        self.eye_pts  = None
        self.detected = False
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, frame):
        with self._lock:
            self._pending = frame

    def _crop_eye(self, frame, cx, cy, size):
        half = size // 2
        h, w = frame.shape[:2]
        x1 = max(0, int(cx-half)); y1 = max(0, int(cy-half))
        x2 = min(w, int(cx+half)); y2 = min(h, int(cy+half))
        crop = frame[y1:y2, x1:x2]
        return crop if (crop.shape[0] >= 8 and crop.shape[1] >= 8) else None

    def _extract(self, crop_bgr):
        rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feat = self._extractor(tensor)
        return feat.squeeze().cpu().numpy()

    def _run(self):
        while self._running:
            frame = None
            with self._lock:
                if self._pending is not None:
                    frame, self._pending = self._pending, None
            if frame is None:
                time.sleep(0.005); continue
            try:
                faces = self._detector(frame)
                if faces is not None and len(faces) > 0:
                    box, landmark, score = faces[0]
                    if score >= 0.5:
                        e0, e1   = landmark[0], landmark[1]
                        face_w   = box[2] - box[0]
                        eye_sz   = max(int(face_w * 0.42), 40)
                        c0 = self._crop_eye(frame, e0[0], e0[1], eye_sz)
                        c1 = self._crop_eye(frame, e1[0], e1[1], eye_sz)
                        if c0 is not None and c1 is not None:
                            feat = np.concatenate([self._extract(c0), self._extract(c1)])
                            with self._lock:
                                self.features = feat
                                self.eye_pts  = ((int(e0[0]), int(e0[1])),
                                                 (int(e1[0]), int(e1[1])))
                                self.detected = True
                            continue
            except Exception:
                pass
            with self._lock:
                self.detected = False

    def read(self):
        with self._lock:
            return self.features, self.eye_pts, self.detected

    def stop(self):
        self._running = False


# ── Camera stream ─────────────────────────────────────────────────────────────
class _WebcamStream:
    def __init__(self, src=0, w=1280, h=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        _, self.frame = self.cap.read()
        self.stopped = False; self.fid = 0

    def start(self):
        threading.Thread(target=self._run, daemon=True).start(); return self

    def _run(self):
        while not self.stopped:
            ok, f = self.cap.read()
            if ok: self.frame = f; self.fid += 1

    def read(self): return self.frame
    def stop(self): self.stopped = True; self.cap.release()


# ── Public API ────────────────────────────────────────────────────────────────
class GazeReader:

    CAL_SAMPLES   = 20
    DRIFT_SAMPLES = 10
    MAX_DRIFT_FIX = 20

    def __init__(self, screen_w=1920, screen_h=1080, device='cpu', cam_src=0):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self._device   = torch.device(device)
        self._cam_src  = cam_src

        self._stream    = None
        self._estimator = None
        self._kalman    = KalmanGaze()

        # Calibration state
        self._cal_features : list = []
        self._cal_targets  : list = []
        self._drift_fixes  : list = []   # (features, target) pairs
        self._cal_ok       = False
        self._cal_x = None
        self._cal_y = None

        # Per-point collection buffer
        self._collecting  = False
        self._col_target  = None
        self._col_buf     : list = []
        self._col_n       = self.CAL_SAMPLES
        self._is_drift    = False

        # Current gaze output
        self._gx = screen_w  // 2
        self._gy = screen_h  // 2
        self._features    = None
        self._eye_pts     = None
        self._face_detected = False
        self._last_fid    = -1

    # ── Lifecycle ─────────────────────────────────────────────────
    def start(self, cam_src=None):
        src = cam_src if cam_src is not None else self._cam_src
        self._stream = _WebcamStream(src=src).start()
        time.sleep(0.5)
        print("Loading ResNet50 (eye-crop mode)…")
        self._estimator = _GazeEstimator(self._device)
        return self

    def update(self):
        """Feed the latest webcam frame into the pipeline. Call once per frame."""
        fid = self._stream.fid
        if fid > self._last_fid:
            frame = self._stream.read()
            if frame is not None:
                self._estimator.submit(frame)
            self._last_fid = fid

        features, eye_pts, detected = self._estimator.read()
        self._face_detected = detected
        self._eye_pts  = eye_pts
        self._features = features if detected else None

        # Update gaze prediction
        if detected and self._cal_ok and features is not None:
            f  = features.reshape(1, -1)
            sx = float(np.clip(self._cal_x.predict(f)[0], 0, 1))
            sy = float(np.clip(self._cal_y.predict(f)[0], 0, 1))
            kx, ky = self._kalman.update(int(sx * self.screen_w),
                                          int(sy * self.screen_h))
            self._gx = int(np.clip(kx, 0, self.screen_w  - 1))
            self._gy = int(np.clip(ky, 0, self.screen_h - 1))

        # Accumulate calibration / drift samples
        if self._collecting and detected and features is not None:
            self._col_buf.append(features.copy())

    def stop(self):
        if self._estimator: self._estimator.stop()
        if self._stream:    self._stream.stop()

    # ── Gaze output ───────────────────────────────────────────────
    def get_gaze(self):
        """Returns (x_px, y_px) in screen pixels."""
        return self._gx, self._gy

    def get_gaze_norm(self):
        """Returns (x, y) normalised to [0, 1]."""
        return self._gx / self.screen_w, self._gy / self.screen_h

    def get_display_frame(self):
        """Mirrored webcam frame with eye markers — ready to show."""
        raw = self._stream.read()
        if raw is None:
            return np.zeros((360, 640, 3), np.uint8)
        cam = cv2.flip(raw, 1)
        if self._eye_pts and self._face_detected:
            iw = cam.shape[1]
            for (ex, ey) in self._eye_pts:
                cv2.circle(cam, (iw - ex, ey), 10, (0, 215, 255), 2)
        cv2.circle(cam, (20, 20), 10,
                   (0, 220, 80) if self._face_detected else (0, 60, 200), -1)
        return cam

    @property
    def face_detected(self):
        return self._face_detected

    # ── Calibration ───────────────────────────────────────────────
    @property
    def is_calibrated(self):
        return self._cal_ok

    def begin_calibration_point(self, norm_x, norm_y, n_samples=None):
        """Start collecting samples for a calibration dot at (norm_x, norm_y)."""
        self._col_target  = (norm_x, norm_y)
        self._col_buf     = []
        self._col_n       = n_samples or self.CAL_SAMPLES
        self._collecting  = True
        self._is_drift    = False

    @property
    def calibration_progress(self):
        """0.0–1.0 progress toward the required sample count."""
        if not self._collecting or self._col_n == 0:
            return 0.0
        return min(len(self._col_buf) / self._col_n, 1.0)

    @property
    def calibration_point_ready(self):
        return self._collecting and len(self._col_buf) >= self._col_n

    def commit_calibration_point(self):
        """Save the collected samples and stop collecting."""
        if self._col_buf:
            avg = np.median(np.array(self._col_buf), axis=0)
            self._cal_features.append(avg)
            self._cal_targets.append(list(self._col_target))
        self._collecting = False
        self._col_buf    = []
        self._col_target = None

    def fit(self):
        """Train the calibration model. Call after all points are committed."""
        X = np.array(self._cal_features)
        Y = np.array(self._cal_targets)
        drift_feat = [d[0] for d in self._drift_fixes]
        drift_tgt  = [d[1] for d in self._drift_fixes]
        if drift_feat:
            X = np.vstack([X, drift_feat])
            Y = np.vstack([Y, drift_tgt])
        n_comp   = min(len(X) - 1, 50)
        self._cal_x = make_pipeline(PCA(n_components=n_comp), Ridge(alpha=10.0))
        self._cal_y = make_pipeline(PCA(n_components=n_comp), Ridge(alpha=10.0))
        self._cal_x.fit(X, Y[:, 0])
        self._cal_y.fit(X, Y[:, 1])
        self._kalman.reset()
        self._cal_ok = True

    def reset_calibration(self):
        self._cal_features.clear()
        self._cal_targets.clear()
        self._drift_fixes.clear()
        self._cal_ok     = False
        self._cal_x      = None
        self._cal_y      = None
        self._collecting = False
        self._col_buf    = []
        self._gx = self.screen_w  // 2
        self._gy = self.screen_h  // 2
        self._kalman.reset()

    # ── Drift correction ──────────────────────────────────────────
    def begin_drift_point(self, norm_x, norm_y, n_samples=None):
        """Start collecting samples for a drift correction dot."""
        self._col_target  = (norm_x, norm_y)
        self._col_buf     = []
        self._col_n       = n_samples or self.DRIFT_SAMPLES
        self._collecting  = True
        self._is_drift    = True

    @property
    def drift_point_ready(self):
        return self._collecting and self._is_drift and len(self._col_buf) >= self._col_n

    def commit_drift_point(self):
        """Save drift fix, refit model."""
        if not self._col_buf:
            return
        avg = np.median(np.array(self._col_buf), axis=0)
        self._drift_fixes.append((avg, list(self._col_target)))
        if len(self._drift_fixes) > self.MAX_DRIFT_FIX:
            self._drift_fixes.pop(0)
        self._collecting = False
        self._col_buf    = []
        self.fit()
