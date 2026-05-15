import cv2
import numpy as np
import time
import mediapipe as mp

from gaze.filters  import OneEuroFilter
from gaze.tracker  import (
    build_detector, get_avg_ear, get_combined_gaze,
    get_head_pose_proxies, get_features,
    fit_calibration, apply_calibration, draw_face_debug,
    IRIS_LEFT, IRIS_RIGHT
)
from gaze.fixation import FixationDetector

# ==========================================
# CONSTANTS
# ==========================================
SCREEN_WIDTH  = 1920
SCREEN_HEIGHT = 1080
BLINK_THRESHOLD = 0.15
SAMPLES_NEEDED  = 30

CALIB_POINTS_NORM = [
    (0.01, 0.01), (0.25, 0.01), (0.5, 0.01), (0.75, 0.01), (0.99, 0.01),
    (0.01, 0.12), (0.5, 0.12),  (0.99, 0.12),
    (0.01, 0.25), (0.5, 0.25),  (0.99, 0.25),
    (0.01, 0.5),  (0.5, 0.5),   (0.99, 0.5),
    (0.01, 0.75), (0.5, 0.75),  (0.99, 0.75),
    (0.01, 0.99), (0.5, 0.99),  (0.99, 0.99),
]

# ==========================================
# STATE
# ==========================================
calib_index            = 0
calib_raw              = []
calib_targets_expanded = []
calib_done             = False
sampling_active        = False
sample_buffer          = []

model_x = model_y = None
ratio_x = ratio_y = 0.0
pitch = yaw = roll = 0.0
features  = (0.0,) * 15
avg_ear   = 0.0
filter_x  = filter_y = None
smooth_x  = smooth_y = 0

fixation_detector = FixationDetector()

# ==========================================
# SETUP
# ==========================================
detector = build_detector('face_landmarker.task')

cv2.namedWindow('Sandbox (Your Screen)', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Sandbox (Your Screen)',
                       cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
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
print("=" * 55)

timestamp_ms = 0

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

            avg_ear = get_avg_ear(landmarks)
            if avg_ear < BLINK_THRESHOLD:
                is_blinking = True

            ratio_x, ratio_y = get_combined_gaze(landmarks)
            pitch, yaw, roll = get_head_pose_proxies(landmarks)
            features         = get_features(landmarks, ratio_x, ratio_y,
                                             pitch, yaw, roll, w, h)
            face_found = True
            draw_face_debug(image, landmarks)

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
            cv2.ellipse(sandbox, (tx, ty), (22, 22), -90, 0, angle,
                        (0, 255, 100), 3)
            cv2.circle(sandbox, (tx, ty), 12, (0, 200, 80), -1)

            status = f"Hold still... {len(sample_buffer)}/{SAMPLES_NEEDED}"
            tsize  = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX,
                                     1.0, 2)[0]
            cv2.putText(sandbox, status,
                        ((SCREEN_WIDTH - tsize[0]) // 2, SCREEN_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

            if len(sample_buffer) >= SAMPLES_NEEDED:
                target = CALIB_POINTS_NORM[calib_index]
                for sample in sample_buffer:
                    calib_raw.append(sample)
                    calib_targets_expanded.append(target)

                print(f"  Point {calib_index+1} captured.")
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

            text  = (f"Look at dot ({calib_index+1}/"
                     f"{len(CALIB_POINTS_NORM)}) naturally — press SPACE")
            tsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                    1.1, 2)[0]
            cv2.putText(sandbox, text,
                        ((SCREEN_WIDTH - tsize[0]) // 2, SCREEN_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 2)

            hint  = "Move your head AND eyes toward the dot naturally"
            hsize = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, 1)[0]
            cv2.putText(sandbox, hint,
                        ((SCREEN_WIDTH - hsize[0]) // 2,
                         SCREEN_HEIGHT // 2 + 60),
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
                cv2.putText(sandbox, "BLINKING",
                            (SCREEN_WIDTH // 2 - 80, 50),
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

                # ---- FIXATION DETECTION ----
                gaze_norm_x = smooth_x / SCREEN_WIDTH
                gaze_norm_y = smooth_y / SCREEN_HEIGHT
                fix_state   = fixation_detector.update(gaze_norm_x,
                                                        gaze_norm_y)

                # Visual feedback on sandbox
                if fix_state['is_confused']:
                    # Red pulsing ring = stuck too long (confused)
                    pulse = int(35 + 10 * abs(np.sin(timestamp_ms / 200)))
                    cv2.circle(sandbox, (smooth_x, smooth_y),
                               pulse, (0, 0, 255), 3)
                    cv2.putText(sandbox, "CONFUSED?",
                                (smooth_x - 60, smooth_y - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2)

                elif fix_state['is_fixation']:
                    # Green ring = stable fixation (reading)
                    dur = fix_state['duration']
                    cv2.circle(sandbox, (smooth_x, smooth_y),
                               32, (0, 200, 0), 2)
                    cv2.putText(sandbox, f"FIX {dur:.1f}s",
                                (smooth_x - 30, smooth_y - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 200, 0), 1)

            # Draw gaze dot
            cv2.circle(sandbox, (smooth_x, smooth_y), 30, (0, 0, 100), -1)
            cv2.circle(sandbox, (smooth_x, smooth_y), 18, (0, 0, 255), -1)
            cv2.circle(sandbox, (smooth_x, smooth_y),  6, (255, 255, 255), -1)

            # Fixation state on camera feed
            fix_label = ("FIXATION" if fix_state['is_fixation']
                         else "SACCADE")
            fix_color = ((0, 255, 0) if fix_state['is_fixation']
                         else (0, 165, 255))
            cv2.putText(image, fix_label, (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, fix_color, 2)

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
        fixation_detector.reset()
        cv2.setWindowProperty('Sandbox (Your Screen)',
                               cv2.WND_PROP_FULLSCREEN,
                               cv2.WINDOW_FULLSCREEN)
        print("Recalibrating...")
    elif key == ord(' ') and not calib_done and not sampling_active:
        sampling_active = True
        sample_buffer.clear()
        print(f"  Sampling point {calib_index+1}... hold still!")

cap.release()
cv2.destroyAllWindows()