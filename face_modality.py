import mediapipe as mp
import threading
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from collections import deque

class FaceModalityTracker:
    def __init__(self, blink_threshold=0.35):
        self.landmark_lock = threading.Lock()
        self.latest_blendshapes = None
        self.new_data_available = False
        self.blink_threshold = blink_threshold
        
        self.is_blinking = False
        self.blink_history = deque(maxlen=900)
        self.fatigue_window = 60.0     # Track blinks over a rolling 60-second window
        self.is_blinking = False
        self.blink_timestamps = deque()
        # NEW: Hysteresis thresholds to ignore squinting
        self.blink_enter_thresh = 0.45 # Eyes must close this much to count as a blink
        self.blink_exit_thresh = 0.25  # Eyes must open this much to reset for the next blink
        self.is_calibrating = False
        self.calib_start_time = None
        self.calib_blink_count = 0
        self.baseline_bpm = 15.0 # Default fallback
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='face_landmarker.task'), 
            num_faces=1, min_face_detection_confidence=0.5, min_tracking_confidence=0.5, 
            output_face_blendshapes=True, running_mode=vision.RunningMode.LIVE_STREAM, 
            output_facial_transformation_matrixes=True,
            result_callback=self._mp_callback)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def _mp_callback(self, result, output_image, timestamp_ms):
            with self.landmark_lock:
                if result.face_blendshapes:
                    self.latest_blendshapes = result.face_blendshapes[0]
                # --- NEW: Grab landmarks for proximity tracking ---
                if result.face_landmarks:
                    self.latest_landmarks = result.face_landmarks[0]
                if result.facial_transformation_matrixes:
                     self.latest_matrix = result.facial_transformation_matrixes[0]
                self.new_data_available = True

    def process_frame(self, image_rgb, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        self.detector.detect_async(mp_image, timestamp_ms)

    # --- NEW BASELINE METHODS ---
    def start_calibration(self):
        self.is_calibrating = True
        self.calib_start_time = time.time()
        self.calib_blink_count = 0
        print("[SYSTEM] Tracking baseline blink rate...")

    def stop_calibration(self):
        self.is_calibrating = False
        if self.calib_start_time:
            duration_sec = time.time() - self.calib_start_time
            duration_min = duration_sec / 60.0
            
            if duration_min > 0:
                raw_bpm = self.calib_blink_count / duration_min
                # Floor the baseline at 15 BPM to prevent the "Staring Contest" effect
                self.baseline_bpm = max(15.0, raw_bpm)
            
            print(f"[SYSTEM] Calibration finished. Baseline BPM set to: {self.baseline_bpm:.1f}")

    def is_fatigued(self):
        """Returns True if current BPM is statistically higher than baseline."""
        if self.is_calibrating:
            return False # Never trigger fatigue while calibrating
            
        current_bpm = len(self.blink_timestamps)
        
        # "Statistically higher" = 50% above baseline AND at least +10 blinks
        fatigue_threshold = max(self.baseline_bpm * 1.5, self.baseline_bpm + 10)
        
        return current_bpm > fatigue_threshold
    # ----------------------------

    def update_state(self):
        process_new_frame = False
        bs = None; lm = None; matrix = None
        
        with self.landmark_lock:
            if self.new_data_available:
                bs = self.latest_blendshapes
                lm = getattr(self, 'latest_landmarks', None)
                matrix = getattr(self, 'latest_matrix', None)
                self.new_data_available = False
                process_new_frame = True

        if process_new_frame and bs is not None:
            blink_l = next((cat.score for cat in bs if cat.category_name == 'eyeBlinkLeft'), 0.0)
            blink_r = next((cat.score for cat in bs if cat.category_name == 'eyeBlinkRight'), 0.0)
            eye_closed_score = (blink_l + blink_r) / 2.0
            
            current_time = time.time()

            # Hysteresis Blink Detection
            if eye_closed_score > self.blink_enter_thresh and not self.is_blinking:
                self.is_blinking = True
                self.blink_timestamps.append(current_time)
                
                # Count blinks specifically for the baseline if calibrating
                if self.is_calibrating:
                    self.calib_blink_count += 1
                    
            elif eye_closed_score < self.blink_exit_thresh and self.is_blinking:
                self.is_blinking = False 

            # Prune blinks older than 60 seconds
            while self.blink_timestamps and (current_time - self.blink_timestamps[0] > self.fatigue_window):
                self.blink_timestamps.popleft()

        return process_new_frame, bs, lm, matrix

    def stop(self):
        self.detector.close()