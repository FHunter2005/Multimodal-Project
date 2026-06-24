"""
Refactored Multimodal Reader Helper Dashboard
====================================================================
Architecture (PURE MATH + ATTENTION FUSION):
- Analyzers: Gaze, Emotion, Epistemic, Mouse, Fused Attention
- Upgrades: 
  1. Real PDF Rasterization (PyMuPDF)
  2. Kalman Filter Restored
  3. Continuous Smooth Pursuit Calibration (Dense Ribbon)
  4. Behavioral Gating (Scroll Kinematics)
"""

import cv2
import time
import math
import numpy as np
import mediapipe as mp
import threading
import pyautogui
import fitz  # NEW: PyMuPDF for handling real PDFs
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from local_epistemic_tracker import LocalEpistemicTracker
from emotion_wheel import EmotionDetector, PlutchikWheel
from mouse_analyzer import MouseReadingAnalyzer
from gaze_analyzer import GazeReadingAnalyzer
from behavior_analyzer import ScrollKinematicsAnalyzer

pyautogui.FAILSAFE = False

# =============================================================
# 1. CORE TRACKER HELPERS & CONSTANTS
# =============================================================
SCREEN_W, SCREEN_H = 1920, 1080
SANDBOX_W, SANDBOX_H, WHEEL_W, WHEEL_H = 1280, 1080, 640, 540

FIXATION_DEADZONE = 0.015  
BLINK_EAR_THRESHOLD = 0.16
SACCADE_VEL_THRESHOLD = 0.8 
SMOOTH_PURSUIT_DURATION = 18.0 

_HEAD_3D = np.array([
    [0.0, 0.0, 0.0],           
    [0.0, -330.0, -65.0],      
    [-225.0, 170.0, -135.0],   
    [225.0, 170.0, -135.0],    
    [-150.0, -150.0, -125.0],  
    [150.0, -150.0, -125.0]    
], dtype=np.float64)
_HEAD_IDX = [1, 152, 33, 263, 61, 291] 

IRIS_LEFT, EYE_LEFT_OUTER, EYE_LEFT_INNER, EYE_LEFT_TOP, EYE_LEFT_BOTTOM = [474, 475, 476, 477], 33, 133, [159, 160, 161], [145, 144, 163]
IRIS_RIGHT, EYE_RIGHT_OUTER, EYE_RIGHT_INNER, EYE_RIGHT_TOP, EYE_RIGHT_BOTTOM = [469, 470, 471, 472], 362, 263, [386, 387, 388], [374, 373, 390]

class KalmanGaze:
    def __init__(self, process_noise=1e-3, measurement_noise=5e-2, dt=1/30):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self._initialized = False
    def update(self, x, y):
        if not self._initialized:
            self.kf.statePre = np.array([[x],[y],[0],[0]], dtype=np.float32); self.kf.statePost = self.kf.statePre.copy(); self._initialized = True; return x, y
        self.kf.predict()
        corrected = self.kf.correct(np.array([[x], [y]], dtype=np.float32))
        return float(corrected[0, 0]), float(corrected[1, 0])
    def reset(self): self._initialized = False

class GazeHeatmap:
    def __init__(self, width=SCREEN_W, height=SCREEN_H, blob_sigma=55.0, decay=0.60, alpha=0.55):
        self.w, self.h, self.decay, self.alpha, self.acc = width, height, decay, alpha, np.zeros((height, width), dtype=np.float32)
        r = int(blob_sigma * 3.5); ax = np.arange(-r, r + 1, dtype=np.float32); xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * blob_sigma**2)); self._kernel, self._r = (kernel / kernel.max()).astype(np.float32), r
    def update(self, gx: int, gy: int):
        self.acc *= self.decay
        x0, x1, y0, y1 = max(gx - self._r, 0), min(gx + self._r + 1, self.w), max(gy - self._r, 0), min(gy + self._r + 1, self.h)
        kx0, ky0 = x0 - (gx - self._r), y0 - (gy - self._r)
        self.acc[y0:y1, x0:x1] += self._kernel[ky0:ky0 + (y1 - y0), kx0:kx0 + (x1 - x0)]
    def render(self, canvas: np.ndarray) -> np.ndarray:
        peak = self.acc.max()
        if peak < 1e-3: return canvas
        norm = np.clip(self.acc / peak, 0.0, 1.0)
        alpha_map = (norm * self.alpha * (norm > 0.05).astype(np.float32))
        color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        for c in range(3): canvas[:, :, c] = np.clip(color[:, :, c] * alpha_map + canvas[:, :, c] * (1.0 - alpha_map), 0, 255).astype(np.uint8)
        return canvas
    def reset(self): self.acc[:] = 0.0

class WebcamVideoStream:
    def __init__(self, src=0, width=1280, height=720):
        self.stream = cv2.VideoCapture(src); self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width); self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.stream.read(); self.stopped, self.frame_id = False, 0
    def start(self): threading.Thread(target=self.update, daemon=True).start(); return self
    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if grabbed: self.frame, self.frame_id = frame, self.frame_id + 1
    def read(self): return self.frame
    def stop(self): self.stopped = True; self.stream.release()

def get_head_rotation_matrix(lm, img_w, img_h):
    pts2d = np.array([(lm[i].x * img_w, lm[i].y * img_h) for i in _HEAD_IDX], dtype=np.float64)
    cam_matrix = np.array([[float(img_w), 0, img_w/2], [0, float(img_w), img_h/2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_HEAD_3D, pts2d, cam_matrix, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return np.eye(3)
    R, _ = cv2.Rodrigues(rvec)
    return R

def unrotate_face(lm, R, img_w, img_h):
    nose_3d = np.array([lm[1].x * img_w, lm[1].y * img_h, lm[1].z * img_w])
    unrotated_lm_2d = []
    for l in lm:
        pt = np.array([l.x * img_w, l.y * img_h, l.z * img_w]) - nose_3d
        u_pt = R.T @ pt  
        unrotated_lm_2d.append((u_pt[0] + nose_3d[0], u_pt[1] + nose_3d[1]))
    return unrotated_lm_2d

def get_eye_gaze_ratio(landmarks, iris_indices, outer, inner, top_ids, bottom_ids):
    iris_pts = [landmarks[i] for i in iris_indices]
    cx, cy = sum(p[0] for p in iris_pts) / len(iris_pts), sum(p[1] for p in iris_pts) / len(iris_pts)
    lx, rx = landmarks[outer][0], landmarks[inner][0]
    
    eye_w = np.hypot(rx - lx, landmarks[inner][1] - landmarks[outer][1])
    eye_h = abs(sum(landmarks[i][1] for i in bottom_ids) / len(bottom_ids) - sum(landmarks[i][1] for i in top_ids) / len(top_ids))
    
    ear = eye_h / (eye_w + 1e-6)
    ratio_x = (cx - lx) / (eye_w + 1e-6)
    ratio_y = (cy - sum(landmarks[i][1] for i in top_ids) / len(top_ids)) / (eye_h + 1e-6)
    
    return ratio_x, ratio_y, ear

# =============================================================
# 2. THE DIALOG CONTROLLER (Data Fusion Logic)
# =============================================================
class DialogController:
    def __init__(self):
        self.struggle_frames = 0
        self.help_active = False
        self.system_message = "Listening to User State..."
        
        self.last_mouse_pos = None
        self.last_mouse_time = time.time()

        self.W_EYE = 0.20
        self.W_HEAD = 0.65
        self.W_VIEWPORT = 0.15
        self.W_MOUSE = 0.00

    def estimate_attention(self, eye_y, mouse_y, head_y, viewport_y, mouse_x):
        current_time = time.time()
        
        if self.last_mouse_pos is None:
            self.last_mouse_pos = (mouse_x, mouse_y)
            
        dist = np.hypot(mouse_x - self.last_mouse_pos[0], mouse_y - self.last_mouse_pos[1])
        self.last_mouse_pos = (mouse_x, mouse_y)
        
        if dist > 3.0:
            self.last_mouse_time = current_time
            
        time_stationary = current_time - self.last_mouse_time
        
        if time_stationary > 1.5:
            target_w_mouse = 0.00
            target_w_eye = 0.20
            target_w_head = 0.65
            target_w_viewport = 0.15
        else:
            target_w_mouse = 0.20
            target_w_eye = 0.25
            target_w_head = 0.45
            target_w_viewport = 0.10
            
        alpha = 0.05
        self.W_MOUSE += (target_w_mouse - self.W_MOUSE) * alpha
        self.W_EYE += (target_w_eye - self.W_EYE) * alpha
        self.W_HEAD += (target_w_head - self.W_HEAD) * alpha
        self.W_VIEWPORT += (target_w_viewport - self.W_VIEWPORT) * alpha

        fused_doc_y = (eye_y * self.W_EYE) + (head_y * self.W_HEAD) + (viewport_y * self.W_VIEWPORT) + (mouse_y * self.W_MOUSE)
        
        # Approximate paragraph index based on typical line heights
        para_idx = int(max(1, (fused_doc_y) // 200 + 1))
        return fused_doc_y, para_idx

    def evaluate(self, mouse_score, gaze_score, epistemic_state, emotion_scores, active_para, is_skimming):
        if is_skimming:
            self.struggle_frames = 0
            self.help_active = False
            return {
                "help_active": False,
                "message": "Skimming Detected. No Intervention Needed.",
                "struggle_level": 0.0,
                "gaze_score": gaze_score,
                "face_score": 0.0,
                "active_para": active_para,
            }

        confusion = epistemic_state.get('confusion', 0.0) if isinstance(epistemic_state, dict) else 0.0
        anger = emotion_scores.get('anger', 0.0) if isinstance(emotion_scores, dict) else 0.0
        sadness = emotion_scores.get('sadness', 0.0) if isinstance(emotion_scores, dict) else 0.0
        
        plutchik_frustration = (anger * 0.70) + (sadness * 0.30)
        face_struggle = max(confusion, plutchik_frustration)

        user_is_stuck = False
        if gaze_score > 0.75: user_is_stuck = True
        elif gaze_score > 0.40 and face_struggle > 0.40: user_is_stuck = True
        elif mouse_score is not None:
            if face_struggle > 0.40 and mouse_score > 0.40: user_is_stuck = True
            elif mouse_score > 0.85: user_is_stuck = True
        elif face_struggle > 0.70: user_is_stuck = True

        if user_is_stuck: self.struggle_frames += 1
        else: self.struggle_frames = max(0, self.struggle_frames - 2)

        TRIGGER_THRESHOLD = 300 
        
        if self.struggle_frames > TRIGGER_THRESHOLD and not self.help_active:
            self.help_active = True
            if plutchik_frustration > confusion: 
                self.system_message = f"INTERVENTION: High Frustration near Section {active_para}. Take a short break."
            elif gaze_score > 0.6: 
                self.system_message = f"INTERVENTION: High Cognitive Load. Simplifying Section {active_para} for you..."
            else: 
                self.system_message = f"INTERVENTION: You seem confused. Generating context tip for Section {active_para}..."
                
        elif self.struggle_frames == 0 and self.help_active:
            self.help_active = False
            self.system_message = "User engaged. Normal reading."

        return {
            "help_active": self.help_active,
            "message": self.system_message,
            "struggle_level": min(1.0, self.struggle_frames / float(TRIGGER_THRESHOLD)),
            "gaze_score": gaze_score,
            "face_score": face_struggle,
            "active_para": active_para,
        }

    def reset_intervention(self):
        self.struggle_frames = 0
        self.help_active = False

# =============================================================
# 3. MAIN APPLICATION CLASS (The Orchestrator)
# =============================================================
class ReaderHelperApp:
    def __init__(self):
        self.SCREEN_W, self.SCREEN_H = SCREEN_W, SCREEN_H
        self.SANDBOX_W, self.SANDBOX_H = SANDBOX_W, SANDBOX_H
        self.WHEEL_W, self.WHEEL_H = WHEEL_W, WHEEL_H
        
        self.vs = WebcamVideoStream(src=0, width=1280, height=720).start()
        self.kalman = KalmanGaze(process_noise=1e-3, measurement_noise=5e-2) 
        self.heatmap = GazeHeatmap()
        self.emotion_detector = EmotionDetector()
        self.emotion_wheel = PlutchikWheel(width=self.WHEEL_W, height=self.WHEEL_H)
        self.epistemic_tracker = LocalEpistemicTracker(window_frames=90, min_frames=20)
        self.mouse_tracker = MouseReadingAnalyzer()
        self.gaze_tracker = GazeReadingAnalyzer(window_size=5.0)
        self.dialog_controller = DialogController()
        self.behavior_analyzer = ScrollKinematicsAnalyzer()
        self.current_gaze_score = 0.0
        self.fused_attention_y = 0.0
        self.active_paragraph = 1
        
        self.inference_mode = False
        self.scroll_y = 0
        
        # Load real PDF or fallback to dummy
        self.active_doc = self._load_real_pdf("document.pdf")
        
        self.calib_done = False
        self.sampling_active = False
        self.calib_start_time = 0
        self.calib_features_eye = []
        self.calib_features_target = []
        
        self.blink_history = deque(maxlen=900)
        self.calib_coeffs_x = None
        self.calib_coeffs_y = None
        self.prev_norm_x = None
        self.prev_norm_y = None

        self.smooth_x, self.smooth_y = self.SCREEN_W // 2, self.SCREEN_H // 2
        self.raw_x_history = deque(maxlen=3) 
        self.raw_y_history = deque(maxlen=3)
        self.raw_eye_x = self.raw_eye_y = 0.5
        self.last_good_eye_x = 0.5
        self.last_good_eye_y = 0.5
        
        self.is_blinking = False
        self.is_saccade = False
        self.last_tracking_time = time.time()

        self.landmark_lock = threading.Lock()
        self.latest_landmarks, self.latest_blendshapes = None, None
        self.new_data_available = False
        
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='face_landmarker.task'), 
            num_faces=1, min_face_detection_confidence=0.5, min_tracking_confidence=0.5, 
            output_face_blendshapes=True, running_mode=vision.RunningMode.LIVE_STREAM, 
            result_callback=self._mp_callback)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        self.mouse_tracker.start()
        self.last_frame_id = -1
        self.display_image = None
        self.running = True

    def _load_real_pdf(self, pdf_path):
        """Rasterizes a real PDF into a continuous vertical NumPy image array."""
        try:
            print(f"Attempting to load real PDF: {pdf_path}...")
            doc = fitz.open(pdf_path)
            pages = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Zoom factor to render text sharply
                zoom = 2.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert fitz pixmap to numpy array
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                
                # Convert RGB to BGR for OpenCV
                if pix.n == 4: # RGBA
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3: # RGB
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                pages.append(img)
            
            if not pages:
                raise Exception("PDF is empty.")
            
            # Pad pages to the max width so we can vstack them
            max_width = max(p.shape[1] for p in pages)
            padded_pages = []
            for p in pages:
                if p.shape[1] < max_width:
                    pad = np.ones((p.shape[0], max_width - p.shape[1], 3), dtype=np.uint8) * 255
                    p = np.hstack((p, pad))
                padded_pages.append(p)
            
            # Stack all pages vertically into one massive image
            full_doc = np.vstack(padded_pages)
            
            # Scale down to fit the UI Sandbox Width (1200 pixels width max)
            target_w = 1200
            scale = target_w / full_doc.shape[1]
            target_h = int(full_doc.shape[0] * scale)
            full_doc = cv2.resize(full_doc, (target_w, target_h))
            
            # Embed the document onto a 1920-wide dark background canvas 
            # so the attention horizon lines match the full screen coordinates
            canvas = np.ones((target_h, self.SCREEN_W, 3), dtype=np.uint8) * 40
            offset_x = (self.SCREEN_W - target_w) // 2
            canvas[:, offset_x:offset_x+target_w] = full_doc
            
            print(f"Successfully loaded {len(doc)} pages.")
            return canvas
            
        except Exception as e:
            print(f"Failed to load {pdf_path}: {e}")
            print("Falling back to dummy generated document.")
            return self._create_dummy_document()

    def _create_dummy_document(self):
        doc = np.ones((3000, 1920, 3), dtype=np.uint8) * 230 
        cv2.rectangle(doc, (360, 50), (1560, 2950), (255, 255, 255), -1)
        cv2.putText(doc, "Chapter 4: The Epistemology of Multimodal AI", (420, 150), cv2.FONT_HERSHEY_DUPLEX, 1.3, (30, 30, 30), 2)
        cv2.line(doc, (420, 170), (1500, 170), (200, 200, 200), 2)
        y_offset = 240
        paragraph = "Multimodal systems process information from diverse sensory channels simultaneously. Unlike unimodal architectures, which might only process text or images in isolation, these systems attempt to achieve late-stage decision level fusion to determine complex internal states such as cognitive load, epistemic confusion, or emotional frustration. "
        for p in range(18):
            cv2.putText(doc, f"[{p+1}] " + paragraph[:80], (420, y_offset), cv2.FONT_HERSHEY_COMPLEX, 0.7, (50, 50, 50), 1)
            cv2.putText(doc, paragraph[80:160], (420, y_offset + 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (50, 50, 50), 1)
            cv2.putText(doc, paragraph[160:], (420, y_offset + 70), cv2.FONT_HERSHEY_COMPLEX, 0.7, (50, 50, 50), 1)
            y_offset += 150
        return doc

    def _mp_callback(self, result, output_image, timestamp_ms):
        with self.landmark_lock:
            if result.face_landmarks:
                self.latest_landmarks = result.face_landmarks[0]
                self.latest_blendshapes = result.face_blendshapes[0]
                self.new_data_available = True

    def _fit_polynomial_calibration(self):
        if len(self.calib_features_eye) < 20:
            print("Not enough calibration points collected. Try again.")
            self.calib_done = self.sampling_active = False
            return

        X = np.array(self.calib_features_eye)
        Y = np.array(self.calib_features_target)
        
        A = np.column_stack([
            np.ones(len(X)), X[:, 0], X[:, 1], X[:, 0]**2, X[:, 1]**2, X[:, 0] * X[:, 1]
        ])
        
        self.calib_coeffs_x, _, _, _ = np.linalg.lstsq(A, Y[:, 0], rcond=None)
        self.calib_coeffs_y, _, _, _ = np.linalg.lstsq(A, Y[:, 1], rcond=None)

        self.calib_done = True
        self.kalman.reset()
        self.heatmap.reset()
        print(f"Calibration Complete: Smooth Pursuit Curve Fitted over {len(X)} frames.")

    def run(self):
        print("Starting Probabilistic Attention Reader Dashboard…")
        time.sleep(1.0)
        cv2.namedWindow('Gaze & Emotion Dashboard', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Gaze & Emotion Dashboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while self.running:
            self._process_camera_feed()
            self._update_and_fuse()
            self._handle_input()
            
        self.cleanup()

    def _process_camera_feed(self):
        current_frame_id = self.vs.frame_id
        if current_frame_id > self.last_frame_id:
            raw_frame = self.vs.read()
            if raw_frame is not None:
                self.display_image = cv2.flip(raw_frame, 1)
                image_rgb = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                self.detector.detect_async(mp_image, int(time.time() * 1000))
                self.last_frame_id = current_frame_id

    def _update_and_fuse(self):
        if self.display_image is None: return

        h, w = self.display_image.shape[:2]
        process_new_frame = False
        sandbox = np.ones((self.SANDBOX_H, self.SANDBOX_W, 3), dtype=np.uint8) * 18
        
        with self.landmark_lock:
            if self.new_data_available:
                lm, bs = self.latest_landmarks, self.latest_blendshapes
                self.new_data_available, process_new_frame = False, True

        mouse_score = self.mouse_tracker.get_score()
        emotion_state = self.emotion_detector.update(bs) if process_new_frame else self.emotion_detector.scores
        
        epistemic_state = {}
        if process_new_frame:
            self.epistemic_tracker.update(bs)
            epistemic_state = getattr(self.epistemic_tracker, 'current_state', {}) 

        head_doc_y = self.scroll_y + (self.SCREEN_H // 2)
        mouse_x, mouse_y = pyautogui.position()
        viewport_doc_y = self.scroll_y + (self.SCREEN_H // 2)
        mouse_doc_y = mouse_y + self.scroll_y

        curr_t = time.time()
        dt = curr_t - self.last_tracking_time

        if process_new_frame and lm:
            R = get_head_rotation_matrix(lm, w, h)
            unrotated_lm_px = unrotate_face(lm, R, w, h)
            
            nose_y_norm = np.clip((lm[1].y - 0.35) / 0.4, 0.0, 1.0) 
            head_screen_y = nose_y_norm * self.SCREEN_H
            head_doc_y = head_screen_y + self.scroll_y

            left_x, left_y, left_ear = get_eye_gaze_ratio(unrotated_lm_px, IRIS_LEFT, EYE_LEFT_OUTER, EYE_LEFT_INNER, EYE_LEFT_TOP, EYE_LEFT_BOTTOM)
            right_x, right_y, right_ear = get_eye_gaze_ratio(unrotated_lm_px, IRIS_RIGHT, EYE_RIGHT_OUTER, EYE_RIGHT_INNER, EYE_RIGHT_TOP, EYE_RIGHT_BOTTOM)

            if left_x is not None and right_x is not None:
                avg_ear = (left_ear + right_ear) / 2.0
                instant_raw_x = (left_x + right_x) / 2.0
                instant_raw_y = (left_y + right_y) / 2.0
                
                if avg_ear > BLINK_EAR_THRESHOLD:
                    self.is_blinking = False
                    self.blink_history.append(0)
                    self.raw_x_history.append(instant_raw_x)
                    self.raw_y_history.append(instant_raw_y)
                    
                    self.raw_eye_x = sum(self.raw_x_history) / len(self.raw_x_history)
                    self.raw_eye_y = sum(self.raw_y_history) / len(self.raw_y_history)
                    
                    # I-VT
                    if dt > 0:
                        eye_velocity = np.hypot(self.raw_eye_x - self.last_good_eye_x, self.raw_eye_y - self.last_good_eye_y) / dt
                        if eye_velocity > SACCADE_VEL_THRESHOLD:
                            self.is_saccade = True
                        else:
                            self.is_saccade = False
                            self.last_good_eye_x = self.raw_eye_x
                            self.last_good_eye_y = self.raw_eye_y
                else:
                    self.is_blinking = True
                    self.is_saccade = False
                    self.blink_history.append(1)
                    self.raw_eye_x = self.last_good_eye_x
                    self.raw_eye_y = self.last_good_eye_y

                self.last_tracking_time = curr_t

                if len(self.blink_history) > 100:
                    perclos_score = sum(self.blink_history) / len(self.blink_history)
                    if perclos_score > 0.15: 
                        self.dialog_controller.system_message = f"INTERVENTION: High Fatigue Detected. Consider a screen break."
                        self.dialog_controller.help_active = True
                
                if self.calib_done:
                    # Freeze coordinates, but safeguard against 'None' on the very first frame
                    if (self.is_blinking or self.is_saccade) and self.prev_norm_x is not None:
                        norm_x, norm_y = self.prev_norm_x, self.prev_norm_y
                    else:
                        smooth_raw_x, smooth_raw_y = self.kalman.update(self.raw_eye_x, self.raw_eye_y)
                        feat = np.array([1, smooth_raw_x, smooth_raw_y, smooth_raw_x**2, smooth_raw_y**2, smooth_raw_x * smooth_raw_y])
                        
                        norm_x = float(np.clip(float(np.dot(feat, self.calib_coeffs_x)), 0.0, 1.0))
                        norm_y = float(np.clip(float(np.dot(feat, self.calib_coeffs_y)), 0.0, 1.0))
                        
                        if self.prev_norm_x is not None:
                            dist_deadzone = np.hypot(norm_x - self.prev_norm_x, norm_y - self.prev_norm_y)
                            if dist_deadzone < FIXATION_DEADZONE:
                                norm_x, norm_y = self.prev_norm_x, self.prev_norm_y
                    
                    self.prev_norm_x, self.prev_norm_y = norm_x, norm_y
                    self.smooth_x = int(norm_x * self.SCREEN_W)
                    self.smooth_y = int(norm_y * self.SCREEN_H)
                    
                    self.heatmap.update(self.smooth_x, self.smooth_y)
                    self.current_gaze_score = self.gaze_tracker.process_point(self.smooth_x, self.smooth_y)
            else:
                self.current_gaze_score = 0.0

        eye_doc_y = self.smooth_y + self.scroll_y
        self.fused_attention_y, self.active_paragraph = self.dialog_controller.estimate_attention(
            eye_y=eye_doc_y, mouse_y=mouse_doc_y, head_y=head_doc_y, viewport_y=viewport_doc_y, mouse_x=mouse_x
        )

        is_skimming = self.behavior_analyzer.analyze(self.scroll_y, dt)

        system_action = self.dialog_controller.evaluate(
            mouse_score, self.current_gaze_score, epistemic_state, emotion_state, self.active_paragraph, is_skimming
        )

        # =============================================================
        # RENDERING
        # =============================================================
        if self.inference_mode and self.calib_done:
            # Clamp scroll to the bounds of the loaded document
            max_scroll = max(0, self.active_doc.shape[0] - self.SCREEN_H)
            self.scroll_y = max(0, min(self.scroll_y, max_scroll))
            
            # Slice the current view from the rasterized PDF
            view = self.active_doc[self.scroll_y : self.scroll_y + self.SCREEN_H, :, :].copy()

            screen_attention_y = int(self.fused_attention_y - self.scroll_y)
            cv2.line(view, (360, screen_attention_y), (1560, screen_attention_y), (0, 150, 255), 2)
            cv2.putText(view, f"FUSED ATTENTION (Section {self.active_paragraph})", (1250, screen_attention_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1)

            if system_action['help_active']:
                cx, cy = self.SCREEN_W // 2, self.SCREEN_H // 2
                cv2.rectangle(view, (cx - 450, cy - 100), (cx + 450, cy + 120), (150, 150, 150), 4)
                cv2.rectangle(view, (cx - 450, cy - 100), (cx + 450, cy + 120), (250, 250, 255), -1)
                cv2.putText(view, "CONTEXTUAL READER HELPER", (cx - 420, cy - 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 100, 200), 2)
                cv2.line(view, (cx - 420, cy - 35), (cx + 420, cy - 35), (200, 200, 200), 2)
                cv2.putText(view, system_action['message'], (cx - 420, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
                cv2.putText(view, "[Press 'C' to clear and continue reading]", (cx - 150, cy + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            cv2.putText(view, "INFERENCE MODE ACTIVE (Real PDF)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
            cv2.putText(view, "[I] Return to Dashboard   [W]/[S] Scroll Document", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            if self.is_blinking:
                cv2.putText(view, "BLINK DETECTED - COORDS FROZEN", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif self.is_saccade:
                cv2.putText(view, "SACCADE DETECTED - COORDS FROZEN", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

            cv2.circle(view, (self.smooth_x, self.smooth_y), 5, (0, 0, 255), -1)
            cv2.imshow('Gaze & Emotion Dashboard', view)

        else:
            if not self.calib_done:
                dashboard = np.ones((self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8) * 18
                self._draw_calibration_ui(dashboard, self.SCREEN_W, self.SCREEN_H)
            else:
                wheel_canvas = self.emotion_wheel.render(emotion_state)
                epistemic_canvas = np.zeros((self.WHEEL_H, self.WHEEL_W, 3), dtype=np.uint8)
                self.epistemic_tracker.render(epistemic_canvas)

                dashboard = np.ones((self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8) * 12
                dashboard[0:self.SANDBOX_H, 0:self.SANDBOX_W] = sandbox                     
                dashboard[0:self.WHEEL_H, self.SANDBOX_W:self.SCREEN_W] = wheel_canvas               
                dashboard[self.WHEEL_H:self.SCREEN_H, self.SANDBOX_W:self.SCREEN_W] = epistemic_canvas        
                
                self._draw_tracking_ui(dashboard, self.SCREEN_W, self.SCREEN_H)
                self._draw_fusion_hud(dashboard, mouse_score, system_action)
                cv2.putText(dashboard, "Press [I] to enter INFERENCE MODE (Reader View)", (1320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Gaze & Emotion Dashboard', dashboard)

    def _draw_calibration_ui(self, canvas, w, h):
        if self.sampling_active:
            elapsed = time.time() - self.calib_start_time
            progress = elapsed / SMOOTH_PURSUIT_DURATION
            
            if elapsed >= SMOOTH_PURSUIT_DURATION:
                self._fit_polynomial_calibration()
                return

            t_curve = elapsed * 1.44
            tx_norm = 0.5 + 0.45 * math.sin(t_curve * 1.13)
            ty_norm = 0.5 + 0.45 * math.sin(t_curve * 1.67 + 0.8) 
            
            tx_norm += 0.03 * math.sin(t_curve * 4.1)
            ty_norm += 0.03 * math.cos(t_curve * 3.7)
            
            tx_norm = max(0.02, min(0.98, tx_norm))
            ty_norm = max(0.02, min(0.98, ty_norm))
            
            tx, ty = int(tx_norm * w), int(ty_norm * h)
            
            if not self.is_blinking and not self.is_saccade:
                self.calib_features_eye.append([self.raw_eye_x, self.raw_eye_y])
                self.calib_features_target.append([tx_norm, ty_norm])

            pulse = int(12 + 4 * abs(np.sin(time.time() * 6)))
            cv2.circle(canvas, (tx, ty), pulse + 6, (0, 255, 100), 3)
            cv2.circle(canvas, (tx, ty), pulse, (0, 200, 80), -1)
            
            cv2.putText(canvas, f"Follow the dot: {int(SMOOTH_PURSUIT_DURATION - elapsed)}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
            cv2.rectangle(canvas, (50, 70), (50 + int(400 * progress), 90), (0, 255, 100), -1)
            cv2.rectangle(canvas, (50, 70), (450, 90), (255, 255, 255), 2)

        else:
            text = "Smooth Pursuit Calibration — Press SPACE"
            text_w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0][0]
            cv2.putText(canvas, text, ((w - text_w) // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
            cv2.putText(canvas, "(Follow the moving green dot closely with your eyes)", ((w - 550) // 2, h // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

    def _draw_tracking_ui(self, canvas, w, h):
        self.heatmap.render(canvas)
        cv2.line(canvas, (self.smooth_x - 14, self.smooth_y), (self.smooth_x + 14, self.smooth_y), (255, 255, 255), 1)
        cv2.line(canvas, (self.smooth_x, self.smooth_y - 14), (self.smooth_x, self.smooth_y + 14), (255, 255, 255), 1)
        cv2.circle(canvas, (self.smooth_x, self.smooth_y), 4, (255, 255, 255), -1)
        
        cv2.putText(canvas, "Tracking Base — 'r' recalibrate  'q' quit", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        status_text = f"Smoothed Ratios: X({self.raw_eye_x:.3f}) Y({self.raw_eye_y:.3f})"
        if self.is_blinking: status_text += " [BLINKING]"
        elif self.is_saccade: status_text += " [SACCADE]"
        cv2.putText(canvas, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255) if self.is_blinking or self.is_saccade else (200,200,200), 1)

    def _draw_fusion_hud(self, canvas, mouse_score, action_state):
        hud_x, hud_y = 30, self.SANDBOX_H - 120
        cv2.rectangle(canvas, (hud_x - 10, hud_y - 40), (hud_x + 550, hud_y + 100), (40, 40, 40), -1)
        
        color = (0, 100, 255) if action_state["help_active"] else (255, 255, 255)
        cv2.putText(canvas, f"SYSTEM: {action_state['message']}", (hud_x, hud_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if mouse_score is not None:
            mouse_text = f"Mouse PAR Score: {mouse_score:.2f}"
            mouse_color = (200, 200, 200)
        else:
            mouse_text = "Mouse PAR Score: INACTIVE"
            mouse_color = (100, 100, 100)
            
        cv2.putText(canvas, mouse_text, (hud_x, hud_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouse_color, 1)
        cv2.putText(canvas, f"Active Section Index: {action_state['active_para']}", (hud_x, hud_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 1)
        cv2.putText(canvas, f"Fusion Struggle Level: {action_state['struggle_level']:.2f}", (hud_x, hud_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    def _handle_input(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            self.running = False
        elif key == ord('i') and self.calib_done:
            self.inference_mode = not self.inference_mode
        elif key == ord('s') and self.inference_mode:
            self.scroll_y += 60
        elif key == ord('w') and self.inference_mode:
            self.scroll_y -= 60
        elif key == ord('c') and self.dialog_controller.help_active:
            self.dialog_controller.reset_intervention()
        elif key == ord('r'):
            self.calib_features_eye.clear()
            self.calib_features_target.clear()
            self.calib_done = self.sampling_active = False
            self.inference_mode = False 
            
            self.raw_x_history.clear()
            self.raw_y_history.clear()
            
            self.smooth_x, self.smooth_y = self.SCREEN_W//2, self.SCREEN_H//2
            self.prev_norm_x = self.prev_norm_y = None
            self.last_good_eye_x = self.last_good_eye_y = 0.5
            self.kalman.reset(); self.heatmap.reset()
            cv2.setWindowProperty('Gaze & Emotion Dashboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif key == ord(' ') and not self.calib_done and not self.sampling_active: 
            self.sampling_active = True
            self.calib_start_time = time.time()
            self.calib_features_eye.clear()
            self.calib_features_target.clear()

    def cleanup(self):
        print("Cleaning up modules...")
        self.vs.stop()
        self.detector.close()
        self.mouse_tracker.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ReaderHelperApp()
    app.run()