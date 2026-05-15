import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# LANDMARK INDICES
# ==========================================
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

NOSE_TIP    = 4
CHIN        = 152
FOREHEAD    = 10
LEFT_CHEEK  = 234
RIGHT_CHEEK = 454

# Brow landmarks for furrow detection (used later in cognitive/)
BROW_LEFT   = [70, 63, 105, 66, 107]
BROW_RIGHT  = [336, 296, 334, 293, 300]


def build_detector(model_path='face_landmarker.task'):
    """Initialize and return the MediaPipe FaceLandmarker detector."""
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.FaceLandmarker.create_from_options(options)


def get_ear(landmarks, outer, inner, top_ids, bottom_ids):
    """
    Eye Aspect Ratio — used for blink detection.
    Low EAR = eye closed/blinking.
    """
    p_left    = landmarks[outer]
    p_right   = landmarks[inner]
    eye_width = math.hypot(p_right[0] - p_left[0], p_right[1] - p_left[1])
    top_y     = sum(landmarks[i][1] for i in top_ids)    / len(top_ids)
    bot_y     = sum(landmarks[i][1] for i in bottom_ids) / len(bottom_ids)
    return abs(bot_y - top_y) / (eye_width + 1e-6)


def get_avg_ear(landmarks):
    """Returns average EAR across both eyes."""
    ear_left  = get_ear(landmarks, EYE_LEFT_OUTER,  EYE_LEFT_INNER,
                         EYE_LEFT_TOP,  EYE_LEFT_BOTTOM)
    ear_right = get_ear(landmarks, EYE_RIGHT_OUTER, EYE_RIGHT_INNER,
                         EYE_RIGHT_TOP, EYE_RIGHT_BOTTOM)
    return (ear_left + ear_right) / 2.0


def get_eye_gaze_ratio(landmarks, iris_indices, outer, inner, top_ids, bottom_ids):
    """
    Returns (ratio_x, ratio_y): iris position relative to eye center,
    normalized by eye width and height. Head-movement invariant.
    """
    iris_pts = [landmarks[i] for i in iris_indices]
    cx = sum(p[0] for p in iris_pts) / len(iris_pts)
    cy = sum(p[1] for p in iris_pts) / len(iris_pts)

    lx, ly = landmarks[outer]
    rx, ry = landmarks[inner]
    eye_w  = np.hypot(rx - lx, ry - ly)

    top_y  = sum(landmarks[i][1] for i in top_ids)    / len(top_ids)
    bot_y  = sum(landmarks[i][1] for i in bottom_ids) / len(bottom_ids)
    eye_h  = abs(bot_y - top_y)

    eye_cx = (lx + rx) / 2
    eye_cy = (top_y + bot_y) / 2

    ratio_x = (cx - eye_cx) / (eye_w + 1e-6)
    ratio_y = (cy - eye_cy) / (eye_h + 1e-6)
    return ratio_x, ratio_y


def get_combined_gaze(landmarks):
    """Average gaze ratio across both eyes for robustness."""
    lx, ly = get_eye_gaze_ratio(landmarks, IRIS_LEFT,
                                  EYE_LEFT_OUTER, EYE_LEFT_INNER,
                                  EYE_LEFT_TOP, EYE_LEFT_BOTTOM)
    rx, ry = get_eye_gaze_ratio(landmarks, IRIS_RIGHT,
                                  EYE_RIGHT_OUTER, EYE_RIGHT_INNER,
                                  EYE_RIGHT_TOP, EYE_RIGHT_BOTTOM)
    return (lx + rx) / 2, (ly + ry) / 2


def get_head_pose_proxies(landmarks):
    """
    Returns (pitch, yaw, roll):
    - pitch: nodding up/down
    - yaw:   turning left/right
    - roll:  tilting head
    """
    nose      = landmarks[NOSE_TIP]
    chin      = landmarks[CHIN]
    left_eye  = landmarks[EYE_LEFT_OUTER]
    right_eye = landmarks[EYE_RIGHT_OUTER]

    dx   = right_eye[0] - left_eye[0]
    dy   = right_eye[1] - left_eye[1]
    roll = math.atan2(dy, dx)

    eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_width    = math.hypot(dx, dy)
    yaw          = (nose[0] - eye_center_x) / (eye_width + 1e-6)

    eye_center_y = (left_eye[1] + right_eye[1]) / 2.0
    face_height  = abs(chin[1] - eye_center_y)
    pitch        = (nose[1] - eye_center_y) / (face_height + 1e-6)

    return pitch, yaw, roll


def get_features(landmarks, ratio_x, ratio_y, pitch, yaw, roll, w, h):
    """
    Full 15-feature vector combining face position, geometry,
    head angles and iris ratios.

    Primary signal: face position in camera frame (nose_x, nose_y)
    — when you look right, your whole head moves right.
    Secondary: head angles (pitch, yaw, roll)
    Fine correction: iris ratios (ratio_x, ratio_y)
    """
    nose     = landmarks[NOSE_TIP]
    chin     = landmarks[CHIN]
    forehead = landmarks[FOREHEAD]
    l_cheek  = landmarks[LEFT_CHEEK]
    r_cheek  = landmarks[RIGHT_CHEEK]

    nose_x           = nose[0] / w
    nose_y           = nose[1] / h
    cheek_spread     = (r_cheek[0] - l_cheek[0]) / w
    face_height_norm = abs(chin[1] - forehead[1]) / h
    l_cheek_x        = l_cheek[0] / w
    r_cheek_x        = r_cheek[0] / w
    face_cx          = (l_cheek[0] + r_cheek[0]) / 2.0 / w
    face_cy          = (forehead[1] + chin[1])    / 2.0 / h
    combined_x       = face_cx + ratio_x * 0.3
    combined_y       = face_cy + ratio_y * 0.3

    return (
        nose_x, nose_y,
        face_cx, face_cy,
        cheek_spread, face_height_norm,
        l_cheek_x, r_cheek_x,
        pitch, yaw, roll,
        ratio_x, ratio_y,
        combined_x, combined_y,
    )


def fit_calibration(raw_samples, target_samples):
    """
    Train two Random Forest models (one for X, one for Y screen coordinate)
    on the calibration data. Returns (model_x, model_y).
    """
    X   = np.array(raw_samples)
    tgt = np.array(target_samples)

    print(f"Training on {len(X)} samples with {X.shape[1]} features...")

    model_x = RandomForestRegressor(
        n_estimators=300, max_depth=12,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    model_y = RandomForestRegressor(
        n_estimators=300, max_depth=12,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    model_x.fit(X, tgt[:, 0])
    model_y.fit(X, tgt[:, 1])

    feat_names = [
        'nose_x', 'nose_y', 'face_cx', 'face_cy',
        'cheek_spread', 'face_height', 'l_cheek_x', 'r_cheek_x',
        'pitch', 'yaw', 'roll',
        'ratio_x', 'ratio_y',
        'combined_x', 'combined_y'
    ]
    print("\nFeature importances (X):")
    for imp, name in sorted(zip(model_x.feature_importances_, feat_names), reverse=True):
        print(f"  {name:15s} {imp:.3f} {'█' * int(imp * 40)}")
    print("\nFeature importances (Y):")
    for imp, name in sorted(zip(model_y.feature_importances_, feat_names), reverse=True):
        print(f"  {name:15s} {imp:.3f} {'█' * int(imp * 40)}")

    return model_x, model_y


def apply_calibration(model_x, model_y, features):
    """Predict screen position (normalized 0-1) from feature vector."""
    X_live = np.array([list(features)])
    pred_x = float(model_x.predict(X_live)[0])
    pred_y = float(model_y.predict(X_live)[0])
    return max(0.0, min(1.0, pred_x)), max(0.0, min(1.0, pred_y))


def draw_face_debug(image, landmarks):
    """Draw iris and key face landmarks on camera feed for debugging."""
    for idx in [IRIS_LEFT, IRIS_RIGHT]:
        pts = [landmarks[i] for i in idx]
        icx = int(sum(pt[0] for pt in pts) / len(pts))
        icy = int(sum(pt[1] for pt in pts) / len(pts))
        cv2.circle(image, (icx, icy), 4, (0, 215, 255), -1)

    for pt_idx in [NOSE_TIP, CHIN, FOREHEAD, LEFT_CHEEK, RIGHT_CHEEK]:
        px = int(landmarks[pt_idx][0])
        py = int(landmarks[pt_idx][1])
        cv2.circle(image, (px, py), 4, (0, 255, 0), -1)