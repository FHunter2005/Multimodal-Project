import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

# Iris landmark indices (468-477 are the iris points added by refine_landmarks)
IRIS_LEFT  = [474, 475, 476, 477]
IRIS_RIGHT = [469, 470, 471, 472]

def draw_landmarks(image, face_landmarks):
    h, w = image.shape[:2]
    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]

    # Draw all mesh points
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    # Draw irises as circles
    for iris_indices in [IRIS_LEFT, IRIS_RIGHT]:
        pts = [landmarks[i] for i in iris_indices]
        cx = int(sum(p[0] for p in pts) / len(pts))
        cy = int(sum(p[1] for p in pts) / len(pts))
        dx = pts[1][0] - pts[3][0]
        dy = pts[1][1] - pts[3][1]
        radius = int(((dx**2 + dy**2) ** 0.5) / 2)
        cv2.circle(image, (cx, cy), radius, (0, 215, 255), 2)

# 2. Open the webcam
cap = cv2.VideoCapture(0)
timestamp_ms = 0

print("Webcam opened! Press 'q' to quit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Cannot read from camera.")
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    timestamp_ms += 33
    result = detector.detect_for_video(mp_image, timestamp_ms)

    if result.face_landmarks:
        for face_landmarks in result.face_landmarks:
            draw_landmarks(image, face_landmarks)

    cv2.imshow('Reader Helper - Camera Test', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()