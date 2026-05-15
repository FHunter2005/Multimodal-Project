import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1 EURO FILTER
# ==========================================
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=0.5, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self.x_prev     = x0
        self.dx_prev    = dx0
        self.t_prev     = t0

    def alpha(self, t_e, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / t_e)

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev
        a_d    = self.alpha(t_e, self.d_cutoff)
        dx     = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a      = self.alpha(t_e, cutoff)
        x_hat  = a * x + (1 - a) * self.x_prev
        self.x_prev  = x_hat
        self.dx_prev = dx_hat
        self.t_prev  = t
        return x_hat

# ==========================================
# MEDIAPIPE SETUP
# ==========================================
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.FaceLandmarker.create_from_options(options)

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

NOSE_TIP         = 4
CHIN             = 152
FOREHEAD         = 10   # top of head landmark
LEFT_CHEEK       = 234  # left face edge
RIGHT_CHEEK      = 454  # right face edge

SCREEN_WIDTH  = 1920
SCREEN_HEIGHT = 1080

CALIB_POINTS_NORM = [
    (0.01, 0.01), (0.25, 0.01), (0.5, 0.01), (0.75, 0.01), (0.99, 0.01),
    (0.01, 0.12), (0.5, 0.12), (0.99, 0.12),
    (0.01, 0.25), (0.5, 0.25), (0.99, 0.25),
    (0.01, 0.5),  (0.5, 0.5),  (0.99, 0.5),
    (0.01, 0.75), (0.5, 0.75), (0.99, 0.75),
    (0.01, 0.99), (0.5, 0.99), (0.99, 0.99),
]

calib_index            = 0
calib_raw              = []
calib_targets_expanded = []
calib_done             = False

# 30 frames ~1 second — just hold still naturally, head AND eyes on the dot
SAMPLES_NEEDED  = 30
sampling_active = False
sample_buffer   = []

BLINK_THRESHOLD = 0.15

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_ear(landmarks, outer, inner, top_ids, bottom_ids):
    p_left    = landmarks[outer]
    p_right   = landmarks[inner]
    eye_width = math.hypot(p_right[0] - p_left[0], p_right[1] - p_left[1])
    top_y     = sum(landmarks[i][1] for i in top_ids)    / len(top_ids)
    bot_y     = sum(landmarks[i][1] for i in bottom_ids) / len(bottom_ids)
    return abs(bot_y - top_y) / (eye_width + 1e-6)


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

    eye_cx = (lx + rx) / 2
    eye_cy = (top_y + bot_y) / 2

    ratio_x = (cx - eye_cx) / (eye_w + 1e-6)
    ratio_y = (cy - eye_cy) / (eye_h + 1e-6)
    return ratio_x, ratio_y


def get_combined_gaze(landmarks):
    lx, ly = get_eye_gaze_ratio(landmarks, IRIS_LEFT,
                                  EYE_LEFT_OUTER, EYE_LEFT_INNER,
                                  EYE_LEFT_TOP, EYE_LEFT_BOTTOM)
    rx, ry = get_eye_gaze_ratio(landmarks, IRIS_RIGHT,
                                  EYE_RIGHT_OUTER, EYE_RIGHT_INNER,
                                  EYE_RIGHT_TOP, EYE_RIGHT_BOTTOM)
    return (lx + rx) / 2, (ly + ry) / 2


def get_head_pose_proxies(landmarks):
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
    Full feature set using the entire face, not just eyes.

    The philosophy:
    - Face position in camera frame (nose_x, nose_y): the PRIMARY signal.
      When you look right, your whole head moves right in the camera.
      This is the strongest and most natural signal.
    - Face geometry (cheek spread, face width): tells the model how far
      you are from the camera, making predictions distance-invariant.
    - Head angles (pitch, yaw, roll): derived rotation, cooperative with gaze.
    - Iris ratios (ratio_x, ratio_y): fine-grained correction on top of head pose.
    - Cooperative combinations: face_x + ratio_x captures the natural
      tendency for eyes and head to move together.
    """

    # --- Primary: Where is your face in the camera frame? ---
    nose      = landmarks[NOSE_TIP]
    chin      = landmarks[CHIN]
    forehead  = landmarks[FOREHEAD]
    l_cheek   = landmarks[LEFT_CHEEK]
    r_cheek   = landmarks[RIGHT_CHEEK]

    # Nose position normalized to camera frame (0=left/top, 1=right/bottom)
    nose_x = nose[0] / w
    nose_y = nose[1] / h

    # Cheek spread: horizontal width of face in frame
    # Captures both face size and left-right head turn
    cheek_spread = (r_cheek[0] - l_cheek[0]) / w

    # Face height in frame: forehead to chin
    face_height_norm = abs(chin[1] - forehead[1]) / h

    # Left/right cheek X positions (captures asymmetry from head turning)
    l_cheek_x = l_cheek[0] / w
    r_cheek_x = r_cheek[0] / w

    # Face center X and Y (average of key landmarks)
    face_cx = (l_cheek[0] + r_cheek[0]) / 2.0 / w
    face_cy = (forehead[1] + chin[1])    / 2.0 / h

    # --- Secondary: Head angles ---
    # These are cooperative with gaze direction

    # --- Fine: Iris position relative to eye ---
    # Small but important for precision within a head-pose region

    # --- Cooperative combinations ---
    # When head and eyes move together (normal behavior), these amplify the signal
    combined_x = face_cx + ratio_x * 0.3
    combined_y = face_cy + ratio_y * 0.3

    return (
        # Face position — strongest signal
        nose_x,           # nose horizontal in camera
        nose_y,           # nose vertical in camera
        face_cx,          # face center horizontal
        face_cy,          # face center vertical

        # Face geometry — distance/size invariance
        cheek_spread,     # face width in frame
        face_height_norm, # face height in frame
        l_cheek_x,        # left cheek position
        r_cheek_x,        # right cheek position

        # Head angles — rotation signals
        pitch,            # nodding
        yaw,              # turning left/right
        roll,             # tilting

        # Iris fine correction
        ratio_x,          # eye horizontal
        ratio_y,          # eye vertical

        # Cooperative combinations
        combined_x,       # face + eye horizontal
        combined_y,       # face + eye vertical
    )


def fit_calibration(raw_samples, target_samples):
    X   = np.array(raw_samples)
    tgt = np.array(target_samples)

    print(f"Training on {len(X)} samples with {X.shape[1]} features...")

    model_x = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model_y = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
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

    print("\nFeature importances (X — horizontal screen):")
    pairs = sorted(zip(model_x.feature_importances_, feat_names), reverse=True)
    for imp, name in pairs:
        bar = '█' * int(imp * 40)
        print(f"  {name:15s} {imp:.3f} {bar}")

    print("\nFeature importances (Y — vertical screen):")
    pairs = sorted(zip(model_y.feature_importances_, feat_names), reverse=True)
    for imp, name in pairs:
        bar = '█' * int(imp * 40)
        print(f"  {name:15s} {imp:.3f} {bar}")

    return model_x, model_y


def apply_calibration(model_x, model_y, features):
    X_live = np.array([list(features)])
    pred_x = float(model_x.predict(X_live)[0])
    pred_y = float(model_y.predict(X_live)[0])
    return max(0.0, min(1.0, pred_x)), max(0.0, min(1.0, pred_y))


# ==========================================
# WINDOW & CAMERA SETUP
# ==========================================
cv2.namedWindow('Sandbox (Your Screen)', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Sandbox (Your Screen)', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.namedWindow('Camera Feed')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("=" * 55)
print("HOW TO CALIBRATE:")
print("  1. Look at the dot NATURALLY — move your head")
print("     AND eyes toward it, just like you normally would")
print("  2. Press SPACE when you're looking at it")
print("  3. Hold completely still for 1 second")
print("  4. Repeat for all 20 dots")
print("  TIP: Sit naturally, don't force head still")
print("  TIP: Look at each dot the way you'd look at")
print("       something on your screen in real use")
print("=" * 55)

timestamp_ms = 0
model_x = model_y = None
ratio_x = ratio_y = 0.0
pitch = yaw = roll = 0.0
features = (0.0,) * 15
avg_ear = 0.0
filter_x = filter_y = None
smooth_x = smooth_y = 0

# ==========================================
# MAIN LOOP
# ==========================================
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image     = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    timestamp_ms += 33
    result = detector.detect_for_video(mp_image, timestamp_ms)

    h, w = image.shape[:2]
    sandbox = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) * 30

    face_found  = False
    is_blinking = False

    if result.face_landmarks:
        for face_landmarks in result.face_landmarks:
            landmarks = [(lm.x * w, lm.y * h) for lm in face_landmarks]

            ear_left  = get_ear(landmarks, EYE_LEFT_OUTER,  EYE_LEFT_INNER,
                                 EYE_LEFT_TOP,  EYE_LEFT_BOTTOM)
            ear_right = get_ear(landmarks, EYE_RIGHT_OUTER, EYE_RIGHT_INNER,
                                 EYE_RIGHT_TOP, EYE_RIGHT_BOTTOM)
            avg_ear   = (ear_left + ear_right) / 2.0

            if avg_ear < BLINK_THRESHOLD:
                is_blinking = True

            ratio_x, ratio_y = get_combined_gaze(landmarks)
            pitch, yaw, roll = get_head_pose_proxies(landmarks)
            features         = get_features(landmarks, ratio_x, ratio_y,
                                             pitch, yaw, roll, w, h)
            face_found = True

            # Draw iris dots
            for idx in [IRIS_LEFT, IRIS_RIGHT]:
                pts = [landmarks[i] for i in idx]
                icx = int(sum(pt[0] for pt in pts) / len(pts))
                icy = int(sum(pt[1] for pt in pts) / len(pts))
                cv2.circle(image, (icx, icy), 4, (0, 215, 255), -1)

            # Draw key face landmarks so you can see what's being tracked
            for pt_idx in [NOSE_TIP, CHIN, FOREHEAD, LEFT_CHEEK, RIGHT_CHEEK]:
                px = int(landmarks[pt_idx][0])
                py = int(landmarks[pt_idx][1])
                cv2.circle(image, (px, py), 4, (0, 255, 0), -1)

    cv2.putText(image, f"EAR: {avg_ear:.3f}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(image,
                f"nose_x:{features[0]:.3f} nose_y:{features[1]:.3f} | "
                f"yaw:{yaw:.2f} pitch:{pitch:.2f}",
                (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ---- CALIBRATION PHASE ----
    if not calib_done:
        tx = int(CALIB_POINTS_NORM[calib_index][0] * SCREEN_WIDTH)
        ty = int(CALIB_POINTS_NORM[calib_index][1] * SCREEN_HEIGHT)

        if sampling_active:
            if face_found and not is_blinking:
                sample_buffer.append(features)

            progress = len(sample_buffer) / SAMPLES_NEEDED
            angle    = int(360 * progress)
            cv2.ellipse(sandbox, (tx, ty), (22, 22), -90, 0, angle, (0, 255, 100), 3)
            cv2.circle(sandbox, (tx, ty), 12, (0, 200, 80), -1)

            status = f"Hold still... {len(sample_buffer)}/{SAMPLES_NEEDED}"
            tsize  = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            cv2.putText(sandbox, status,
                        ((SCREEN_WIDTH - tsize[0]) // 2, SCREEN_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

            if len(sample_buffer) >= SAMPLES_NEEDED:
                target = CALIB_POINTS_NORM[calib_index]
                for sample in sample_buffer:
                    calib_raw.append(sample)
                    calib_targets_expanded.append(target)

                print(f"  Point {calib_index+1} captured ({len(sample_buffer)} samples).")
                sample_buffer.clear()
                sampling_active = False
                calib_index    += 1

                if calib_index == len(CALIB_POINTS_NORM):
                    print("Fitting Random Forest... please wait.")
                    model_x, model_y = fit_calibration(
                        calib_raw, calib_targets_expanded)
                    calib_done = True
                    print("Done! Tracking active.")

        else:
            pulse = int(10 + 6 * abs(np.sin(timestamp_ms / 300)))
            cv2.circle(sandbox, (tx, ty), pulse + 6, (255, 255, 255), 2)
            cv2.circle(sandbox, (tx, ty), pulse, (0, 0, 220), -1)

            text  = f"Look at dot ({calib_index+1}/{len(CALIB_POINTS_NORM)}) naturally — press SPACE"
            tsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)[0]
            cv2.putText(sandbox, text,
                        ((SCREEN_WIDTH - tsize[0]) // 2, SCREEN_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 2)

            hint = "Move your head AND eyes toward the dot, as you naturally would"
            hsize = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
            cv2.putText(sandbox, hint,
                        ((SCREEN_WIDTH - hsize[0]) // 2, SCREEN_HEIGHT // 2 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 1)

        for i, (px, py) in enumerate(CALIB_POINTS_NORM):
            if i < calib_index:
                color, size = (0, 200, 0), 8
            elif i == calib_index:
                color, size = (255, 255, 255), 5
            else:
                color, size = (80, 80, 80), 6
            cv2.circle(sandbox,
                       (int(px * SCREEN_WIDTH), int(py * SCREEN_HEIGHT)),
                       size, color, -1)

    # ---- TRACKING PHASE ----
    else:
        if face_found:
            if is_blinking:
                cv2.putText(sandbox, "BLINKING (Cursor Frozen)",
                            (SCREEN_WIDTH // 2 - 200, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                sx, sy = apply_calibration(model_x, model_y, features)
                raw_px = int(sx * SCREEN_WIDTH)
                raw_py = int(sy * SCREEN_HEIGHT)

                curr_time = time.time()

                if filter_x is None:
                    filter_x = OneEuroFilter(curr_time, raw_px,
                                             min_cutoff=0.5, beta=0.05)
                    filter_y = OneEuroFilter(curr_time, raw_py,
                                             min_cutoff=0.5, beta=0.05)

                smooth_x = int(filter_x(curr_time, raw_px))
                smooth_y = int(filter_y(curr_time, raw_py))

                smooth_x = max(0, min(SCREEN_WIDTH  - 1, smooth_x))
                smooth_y = max(0, min(SCREEN_HEIGHT - 1, smooth_y))

            cv2.circle(sandbox, (smooth_x, smooth_y), 30, (0, 0, 100), -1)
            cv2.circle(sandbox, (smooth_x, smooth_y), 18, (0, 0, 255), -1)
            cv2.circle(sandbox, (smooth_x, smooth_y),  6, (255, 255, 255), -1)

        cv2.putText(sandbox, "Press 'r' to recalibrate  |  'q' to quit",
                    (20, SCREEN_HEIGHT - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 1)
        cv2.putText(image, "RF Tracking Active! Press 'r' to recalibrate",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Sandbox (Your Screen)', sandbox)
    cv2.imshow('Camera Feed', image)

    key = cv2.waitKey(5) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        calib_index            = 0
        calib_raw              = []
        calib_targets_expanded = []
        calib_done             = False
        sampling_active        = False
        sample_buffer.clear()
        model_x = model_y      = None
        filter_x = filter_y    = None
        smooth_x = smooth_y    = 0
        cv2.setWindowProperty('Sandbox (Your Screen)',
                               cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        print("Recalibrating...")
    elif key == ord(' ') and not calib_done and not sampling_active:
        sampling_active = True
        sample_buffer.clear()
        print(f"  Sampling point {calib_index+1}... hold still!")

cap.release()
cv2.destroyAllWindows()