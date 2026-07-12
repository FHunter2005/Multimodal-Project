import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque 

base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.FaceLandmarker.create_from_options(options)

IRIS_LEFT  = [474, 475, 476, 477]

# Screen dimensions
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Calibration Variables
calibration_state = 0 
tl_cx, tl_cy = 0, 0
br_cx, br_cy = 0, 0

SMOOTHING_FRAMES = 10
history_x = deque(maxlen=SMOOTHING_FRAMES)
history_y = deque(maxlen=SMOOTHING_FRAMES)

def get_pupil_center(landmarks, iris_indices):
    pts = [landmarks[i] for i in iris_indices]
    cx = int(sum(p[0] for p in pts) / len(pts))
    cy = int(sum(p[1] for p in pts) / len(pts))
    return cx, cy

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Webcam opened in HD! Look at the popup window.")

timestamp_ms = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    timestamp_ms += 33
    result = detector.detect_for_video(mp_image, timestamp_ms)

    h, w = image.shape[:2]

    if result.face_landmarks:
        for face_landmarks in result.face_landmarks:
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]
            
            cx, cy = get_pupil_center(landmarks, IRIS_LEFT)
            cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1)

            if calibration_state == 0:
                cv2.putText(image, "Look TOP-LEFT and press '1'", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            elif calibration_state == 1:
                cv2.putText(image, "Look BOTTOM-RIGHT and press '2'", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            elif calibration_state == 2:
                try:

                    raw_screen_x = int((cx - tl_cx) / (br_cx - tl_cx) * SCREEN_WIDTH)
                    raw_screen_y = int((cy - tl_cy) / (br_cy - tl_cy) * SCREEN_HEIGHT)
                    
                    raw_screen_x = max(0, min(SCREEN_WIDTH, raw_screen_x))
                    raw_screen_y = max(0, min(SCREEN_HEIGHT, raw_screen_y))

                    history_x.append(raw_screen_x)
                    history_y.append(raw_screen_y)

                    smooth_x = int(sum(history_x) / len(history_x))
                    smooth_y = int(sum(history_y) / len(history_y))

                    cv2.putText(image, f"Looking at: X:{smooth_x} Y:{smooth_y}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except ZeroDivisionError:
                    calibration_state = 0 

    cv2.imshow('Reader Helper - Calibration', image)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1') and calibration_state == 0:
        tl_cx, tl_cy = cx, cy
        print(f"HD Top-Left saved: {tl_cx}, {tl_cy}")
        calibration_state = 1
    elif key == ord('2') and calibration_state == 1:
        br_cx, br_cy = cx, cy
        print(f"HD Bottom-Right saved: {br_cx}, {br_cy}")
        calibration_state = 2

cap.release()
cv2.destroyAllWindows()