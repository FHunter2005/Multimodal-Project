import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# 1. Initialize MediaPipe Face Landmarker
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.FaceLandmarker.create_from_options(options)

# --- Landmark indices ---
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

# --- Calibration: 9-point grid ---
CALIB_POINTS_NORM = [
    (0.02, 0.02), (0.5, 0.02), (0.98, 0.02),
    (0.02, 0.25), (0.5, 0.25), (0.98, 0.25),
    (0.02, 0.5),  (0.5, 0.5),  (0.98, 0.5),
    (0.02, 0.75), (0.5, 0.75), (0.98, 0.75),
    (0.02, 0.98), (0.5, 0.98), (0.98, 0.98),
]
calib_index = 0
calib_raw   = []
calib_done  = False

SMOOTHING_FRAMES = 8
history_x = deque(maxlen=SMOOTHING_FRAMES)
history_y = deque(maxlen=SMOOTHING_FRAMES)


def get_eye_gaze_ratio(landmarks, iris_indices,
                        outer, inner, top_ids, bottom_ids):
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


def fit_calibration(calib_raw, calib_targets):
    raw  = np.array(calib_raw)
    tgt  = np.array(calib_targets)
    ones = np.ones((len(raw), 1))
    A    = np.hstack([raw, ones])
    cx, _, _, _ = np.linalg.lstsq(A, tgt[:, 0], rcond=None)
    cy, _, _, _ = np.linalg.lstsq(A, tgt[:, 1], rcond=None)
    return cx, cy


def apply_calibration(cx, cy, raw_x, raw_y):
    v        = np.array([raw_x, raw_y, 1.0])
    screen_x = float(np.dot(cx, v))
    screen_y = float(np.dot(cy, v))
    screen_x = max(0.0, min(1.0, screen_x))
    screen_y = max(0.0, min(1.0, screen_y))
    return screen_x, screen_y


# --- Setup windows ---
cv2.namedWindow('Sandbox (Your Screen)', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Sandbox (Your Screen)', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.namedWindow('Camera Feed')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Webcam opened!")
print("Follow the red dot and press SPACE to capture each calibration point.")

timestamp_ms = 0
cx_coeff = cy_coeff = None
ratio_x = ratio_y = 0.0

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

    # Always create sandbox at full screen resolution
    sandbox = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) * 30

    if result.face_landmarks:
        for face_landmarks in result.face_landmarks:
            landmarks = [(lm.x * w, lm.y * h) for lm in face_landmarks]
            ratio_x, ratio_y = get_combined_gaze(landmarks)

            # Draw iris centers on camera feed
            for idx in [IRIS_LEFT, IRIS_RIGHT]:
                pts  = [landmarks[i] for i in idx]
                icx  = int(sum(p[0] for p in pts) / len(pts))
                icy  = int(sum(p[1] for p in pts) / len(pts))
                cv2.circle(image, (icx, icy), 4, (0, 215, 255), -1)

    # ---- Calibration phase ----
    if not calib_done:
        tx = int(CALIB_POINTS_NORM[calib_index][0] * SCREEN_WIDTH)
        ty = int(CALIB_POINTS_NORM[calib_index][1] * SCREEN_HEIGHT)

        # Pulsing dot
        pulse = int(10 + 6 * abs(np.sin(timestamp_ms / 300)))
        cv2.circle(sandbox, (tx, ty), pulse + 6, (255, 255, 255), 2)
        cv2.circle(sandbox, (tx, ty), pulse, (0, 0, 220), -1)

        # Instruction text centered on screen
        text  = f"Look at the dot ({calib_index+1}/9) and press SPACE"
        tsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        tx_   = (SCREEN_WIDTH - tsize[0]) // 2
        cv2.putText(sandbox, text, (tx_, SCREEN_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)

        # Progress dots
        for i, (px, py) in enumerate(CALIB_POINTS_NORM):
            color = (0, 200, 0) if i < calib_index else (100, 100, 100)
            cv2.circle(sandbox,
                       (int(px * SCREEN_WIDTH), int(py * SCREEN_HEIGHT)),
                       8, color, -1)

    # ---- Tracking phase ----
    else:
        sx, sy   = apply_calibration(cx_coeff, cy_coeff, ratio_x, ratio_y)
        raw_px   = int(sx * SCREEN_WIDTH)
        raw_py   = int(sy * SCREEN_HEIGHT)

        history_x.append(raw_px)
        history_y.append(raw_py)

        weights  = np.linspace(0.5, 1.0, len(history_x))
        smooth_x = int(np.average(list(history_x), weights=weights))
        smooth_y = int(np.average(list(history_y), weights=weights))

        # Gaze dot with glow effect
        cv2.circle(sandbox, (smooth_x, smooth_y), 30, (0, 0, 100), -1)
        cv2.circle(sandbox, (smooth_x, smooth_y), 18, (0, 0, 255), -1)
        cv2.circle(sandbox, (smooth_x, smooth_y), 6,  (255, 255, 255), -1)

        cv2.putText(sandbox, "Press 'r' to recalibrate  |  'q' to quit",
                    (20, SCREEN_HEIGHT - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 1)

        cv2.putText(image, "Tracking Active! Press 'r' to recalibrate",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Gaze  x:{ratio_x:.3f}  y:{ratio_y:.3f}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow('Sandbox (Your Screen)', sandbox)
    cv2.imshow('Camera Feed', image)

    key = cv2.waitKey(5) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('r'):
        calib_index = 0
        calib_raw   = []
        calib_done  = False
        history_x.clear()
        history_y.clear()
        cv2.setWindowProperty('Sandbox (Your Screen)',
                               cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        print("Recalibrating...")

    elif key == ord(' ') and not calib_done:
        calib_raw.append((ratio_x, ratio_y))
        print(f"  Point {calib_index+1} captured: ({ratio_x:.4f}, {ratio_y:.4f})")
        calib_index += 1

        if calib_index == len(CALIB_POINTS_NORM):
            cx_coeff, cy_coeff = fit_calibration(calib_raw, CALIB_POINTS_NORM)
            calib_done = True
            print("Calibration complete! Tracking active.")

cap.release()
cv2.destroyAllWindows()