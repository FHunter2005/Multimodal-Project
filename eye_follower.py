import cv2
import time
import numpy as np
import mediapipe as mp
import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Create two global ML models
model_x = make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.02, kernel='rbf', gamma='scale'))
model_y = make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.02, kernel='rbf', gamma='scale'))

def fit_calibration(calib_raw, calib_targets):
    X = np.array(calib_raw)
    Y = np.array(calib_targets)
    
    # Train the machine learning models
    model_x.fit(X, Y[:, 0])
    model_y.fit(X, Y[:, 1])
    return True

def apply_calibration(raw_x, raw_y):
    # Predict screen coordinates based on gaze ratio
    pred = np.array([[raw_x, raw_y]])
    screen_x = model_x.predict(pred)[0]
    screen_y = model_y.predict(pred)[0]
    return np.clip(screen_x, 0.0, 1.0), np.clip(screen_y, 0.0, 1.0)
# ---------------------------------------------------------
# Threaded Camera Class for Non-Blocking I/O
# ---------------------------------------------------------
class WebcamVideoStream:
    def __init__(self, src=0, width=1280, height=720):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.frame_id = 0

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (grabbed, frame) = self.stream.read()
            if grabbed:
                self.frame = frame
                self.frame_id += 1 

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# ---------------------------------------------------------
# Constants & Setup
# ---------------------------------------------------------
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
    # Center Anchor
    (0.50, 0.50), 
    
    # Inner Diamond (For core focus accuracy)
    (0.50, 0.30), (0.50, 0.70), (0.35, 0.50), (0.65, 0.50),
    
    # Outer Ring Corners
    (0.05, 0.05), (0.95, 0.05), (0.05, 0.95), (0.95, 0.95),
    
    # Ultimate Edge midpoints (Crucial for screen-edge tracking)
    (0.50, 0.01), (0.50, 0.99), (0.01, 0.50), (0.99, 0.50),
    
    # Intermediate Filler Points
    (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)
]
calib_index   = 0
calib_raw     = []
calib_done    = False

SAMPLES_NEEDED   = 30
sampling_active  = False
sample_buffer    = []

# ---------------------------------------------------------
# MediaPipe Async Setup
# ---------------------------------------------------------
latest_landmarks = None
landmark_lock = threading.Lock()
new_data_available = False 

def result_callback(result, output_image, timestamp_ms):
    global latest_landmarks, new_data_available
    with landmark_lock:
        if result.face_landmarks:
            latest_landmarks = result.face_landmarks[0]
        else:
            latest_landmarks = None
        new_data_available = True 

base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=result_callback
)
detector = vision.FaceLandmarker.create_from_options(options)

# ---------------------------------------------------------
# Math & Tracking Functions
# ---------------------------------------------------------
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
    
    # ---- BLINK REJECTION ----
    ear = eye_h / (eye_w + 1e-6)
    if ear < 0.18: # Threshold for a closed eye
        return None, None

    eye_cx = (lx + rx) / 2
    eye_cy = (top_y + bot_y) / 2

    ratio_x = (cx - eye_cx) / (eye_w + 1e-6)
    ratio_y = (cy - eye_cy) / (eye_h + 1e-6)
    return ratio_x, ratio_y

def get_combined_gaze(landmarks):
    lx, ly = get_eye_gaze_ratio(landmarks, IRIS_LEFT, EYE_LEFT_OUTER, EYE_LEFT_INNER, EYE_LEFT_TOP, EYE_LEFT_BOTTOM)
    rx, ry = get_eye_gaze_ratio(landmarks, IRIS_RIGHT, EYE_RIGHT_OUTER, EYE_RIGHT_INNER, EYE_RIGHT_TOP, EYE_RIGHT_BOTTOM)
    
    # If either eye is blinking, reject the frame
    if lx is None or rx is None:
        return None, None
        
    return (lx + rx) / 2, (ly + ry) / 2

def poly_features(x, y):
    return np.array([1, x, y, x*x, y*y, x*y])


# ---------------------------------------------------------
# Main Application
# ---------------------------------------------------------
cv2.namedWindow('Sandbox (Your Screen)', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Sandbox (Your Screen)', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.namedWindow('Camera Feed')

print("Starting camera thread...")
vs = WebcamVideoStream(src=0, width=1280, height=720).start()
time.sleep(1.0) 

print("Webcam opened! Look at each dot, press SPACE, then HOLD STILL for 1 second.")

cx_coeff = cy_coeff = None
ratio_x = ratio_y = 0.0
smooth_x = smooth_y = SCREEN_WIDTH // 2 
current_landmarks = None 

# EMA Smoothing variables
ALPHA_STILL = 0.1  # Heavy smoothing when staring
ALPHA_MOVE  = 0.6  # Light smoothing when tracking rapid movement

last_frame_id = -1
display_image = None

while True:
    current_frame_id = vs.frame_id
    sandbox = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) * 30
    
    # 1. Send frame to MediaPipe
    if current_frame_id > last_frame_id:
        raw_frame = vs.read()
        if raw_frame is None:
            continue
            
        image = cv2.flip(raw_frame, 1)
        display_image = image.copy() 
        
        current_timestamp_ms = int(time.time() * 1000)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        detector.detect_async(mp_image, current_timestamp_ms)
        last_frame_id = current_frame_id
    else:
        if display_image is None: continue
        image = display_image.copy()

    h, w = image.shape[:2]

    # 2. Check for fresh data safely
    process_new_frame = False
    with landmark_lock:
        if new_data_available:
            current_landmarks = latest_landmarks
            new_data_available = False 
            process_new_frame = True
            
    # 3. ONLY update math and smoothing if we actually got a new ML result
    face_found = False
    if process_new_frame and current_landmarks:
        face_found = True
        landmarks = [(lm.x * w, lm.y * h) for lm in current_landmarks]
        
        # Returns None, None if blinking
        new_gaze_x, new_gaze_y = get_combined_gaze(landmarks)
        
        # Only update mathematical tracking if eyes are OPEN
        if new_gaze_x is not None:
            ratio_x, ratio_y = new_gaze_x, new_gaze_y

            # Apply tracking cursors
                        # Apply tracking cursors
            if calib_done:
                sx, sy = apply_calibration(ratio_x, ratio_y) # Remove the old cx_coeff and cy_coeff
                raw_px = int(sx * SCREEN_WIDTH)
                raw_py = int(sy * SCREEN_HEIGHT)

                # ---- EMA SMOOTHING LOGIC ----
                distance = np.hypot(raw_px - smooth_x, raw_py - smooth_y)
                current_alpha = ALPHA_STILL if distance < 50 else ALPHA_MOVE
                
                smooth_x = int(current_alpha * raw_px + (1.0 - current_alpha) * smooth_x)
                smooth_y = int(current_alpha * raw_py + (1.0 - current_alpha) * smooth_y)

    # --- Draw UI (happens continuously, using the smoothed coordinates) ---
    if current_landmarks:
         landmarks_draw = [(lm.x * w, lm.y * h) for lm in current_landmarks]
         for idx in [IRIS_LEFT, IRIS_RIGHT]:
            pts = [landmarks_draw[i] for i in idx]
            icx = int(sum(p[0] for p in pts) / len(pts))
            icy = int(sum(p[1] for p in pts) / len(pts))
            cv2.circle(image, (icx, icy), 4, (0, 215, 255), -1)

    # ---- Calibration Phase ----
    if not calib_done:
        tx = int(CALIB_POINTS_NORM[calib_index][0] * SCREEN_WIDTH)
        ty = int(CALIB_POINTS_NORM[calib_index][1] * SCREEN_HEIGHT)

        if sampling_active:
            # Note: ratio_x/y only updates if eyes are open, so blinks don't corrupt calibration
            if face_found: 
                sample_buffer.append((ratio_x, ratio_y))

            progress = len(sample_buffer) / SAMPLES_NEEDED
            angle    = int(360 * progress)

            cv2.ellipse(sandbox, (tx, ty), (22, 22), -90, 0, angle, (0, 255, 100), 3)
            cv2.circle(sandbox, (tx, ty), 12, (0, 200, 80), -1)

            cv2.putText(sandbox, f"Hold still... {len(sample_buffer)}/{SAMPLES_NEEDED}",
                        ((SCREEN_WIDTH - 400) // 2, SCREEN_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

            if len(sample_buffer) >= SAMPLES_NEEDED:
                avg_x = sum(s[0] for s in sample_buffer) / len(sample_buffer)
                avg_y = sum(s[1] for s in sample_buffer) / len(sample_buffer)
                calib_raw.append((avg_x, avg_y))
                print(f"Point {calib_index+1} captured: ({avg_x:.4f}, {avg_y:.4f})")

                sample_buffer.clear()
                sampling_active = False
                calib_index    += 1

                if calib_index == len(CALIB_POINTS_NORM):
                    fit_calibration(calib_raw, CALIB_POINTS_NORM) # Just call it without assigning variables
                    calib_done = True
                    print("Calibration complete! Tracking active.")
        else:
            pulse = int(10 + 6 * abs(np.sin(time.time() * 3)))
            cv2.circle(sandbox, (tx, ty), pulse + 6, (255, 255, 255), 2)
            cv2.circle(sandbox, (tx, ty), pulse, (0, 0, 220), -1)

            text  = f"Look at dot ({calib_index+1}/{len(CALIB_POINTS_NORM)}) — press SPACE"
            tsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            cv2.putText(sandbox, text, ((SCREEN_WIDTH - tsize[0]) // 2, SCREEN_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)

        for i, (px, py) in enumerate(CALIB_POINTS_NORM):
            if i < calib_index:   color, size = (0, 200, 0), 8
            elif i == calib_index: color, size = (255, 255, 255), 5
            else:                  color, size = (80, 80, 80), 6
            cv2.circle(sandbox, (int(px * SCREEN_WIDTH), int(py * SCREEN_HEIGHT)), size, color, -1)

    # ---- Tracking Phase ----
    else:
        cv2.circle(sandbox, (smooth_x, smooth_y), 30, (0, 0, 100), -1)
        cv2.circle(sandbox, (smooth_x, smooth_y), 18, (0, 0, 255), -1)
        cv2.circle(sandbox, (smooth_x, smooth_y), 6,  (255, 255, 255), -1)

        cv2.putText(sandbox, "Press 'r' to recalibrate  |  'q' to quit",
                    (20, SCREEN_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 1)
        cv2.putText(image, "Tracking Active! Press 'r' to recalibrate",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Gaze  x:{ratio_x:.3f}  y:{ratio_y:.3f}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow('Sandbox (Your Screen)', sandbox)
    cv2.imshow('Camera Feed', image)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        calib_index = 0
        calib_raw = []
        calib_done = False
        sampling_active = False
        sample_buffer.clear()
        smooth_x = smooth_y = SCREEN_WIDTH // 2 
        cv2.setWindowProperty('Sandbox (Your Screen)', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        print("Recalibrating...")
    elif key == ord(' ') and not calib_done and not sampling_active:
        sampling_active = True
        sample_buffer.clear()
        print(f"Sampling point {calib_index+1}... hold still!")

vs.stop()
detector.close()
cv2.destroyAllWindows()