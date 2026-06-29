import mediapipe as mp
import threading
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceModalityTracker:
    def __init__(self, blink_threshold=0.35):
        self.landmark_lock = threading.Lock()
        self.latest_blendshapes = None
        self.new_data_available = False
        self.blink_threshold = blink_threshold
        
        self.is_blinking = False
        self.blink_history = deque(maxlen=900)

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='face_landmarker.task'), 
            num_faces=1, min_face_detection_confidence=0.5, min_tracking_confidence=0.5, 
            output_face_blendshapes=True, running_mode=vision.RunningMode.LIVE_STREAM, 
            result_callback=self._mp_callback)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def _mp_callback(self, result, output_image, timestamp_ms):
            with self.landmark_lock:
                if result.face_blendshapes:
                    self.latest_blendshapes = result.face_blendshapes[0]
                # --- NEW: Grab landmarks for proximity tracking ---
                if result.face_landmarks:
                    self.latest_landmarks = result.face_landmarks[0]
                self.new_data_available = True

    def process_frame(self, image_rgb, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        self.detector.detect_async(mp_image, timestamp_ms)

    def update_state(self):
        process_new_frame = False
        bs = None
        lm = None  # <--- Added lm

        with self.landmark_lock:
            if self.new_data_available:
                bs = self.latest_blendshapes
                lm = getattr(self, 'latest_landmarks', None) # <--- Extract landmarks safely
                self.new_data_available = False
                process_new_frame = True

        if process_new_frame and bs is not None:
            blink_l = next((cat.score for cat in bs if cat.category_name == 'eyeBlinkLeft'), 0.0)
            blink_r = next((cat.score for cat in bs if cat.category_name == 'eyeBlinkRight'), 0.0)
            if (blink_l + blink_r) / 2.0 > self.blink_threshold:
                self.is_blinking = True
                self.blink_history.append(1)
            else:
                self.is_blinking = False
                self.blink_history.append(0)

        return process_new_frame, bs, lm # <--- Return all three

    def get_fatigue_score(self):
        if len(self.blink_history) > 100:
            return sum(self.blink_history) / len(self.blink_history)
        return 0.0

    def stop(self):
        self.detector.close()