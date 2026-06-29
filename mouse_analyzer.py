import time
import math
from collections import deque, Counter # <-- Added Counter
from pynput import mouse

class MouseReadingAnalyzer:
    def __init__(self, short_window=4.0, long_window=10.0, state_buffer_size=30):
        self.short_window = short_window
        self.long_window = long_window
        self.history = deque()
        
        self.is_active_reader = False
        self.smoothed_intensity = 0.0
        self._listener = None
        
        # NEW: A buffer to hold recent states for majority voting
        self.state_buffer = deque(maxlen=state_buffer_size) 

    def start(self):
        self._listener = mouse.Listener(on_move=self._on_move)
        self._listener.start()
        print("[INFO] Mouse Modality Started (Kinematic State Detection).")

    def stop(self):
        if self._listener:
            self._listener.stop()

    def _on_move(self, x, y):
        self.history.append((time.time(), x, y))

    def _calc_kinematics(self, points):
        # ... (Keep your existing _calc_kinematics exactly as it is) ...
        if len(points) < 2:
            return 0.0, 0.0, 0.0, 0
            
        dist, x_dist, y_dist = 0.0, 0.0, 0.0
        x_flips, y_flips = 0, 0
        last_dx_sign, last_dy_sign = 0, 0

        for i in range(1, len(points)):
            dx = points[i][1] - points[i-1][1]
            dy = points[i][2] - points[i-1][2]
            
            dist += math.sqrt(dx**2 + dy**2)
            x_dist += abs(dx)
            y_dist += abs(dy)
            
            if abs(dx) > 2.0:
                dx_sign = 1 if dx > 0 else -1
                if last_dx_sign != 0 and dx_sign != last_dx_sign: x_flips += 1
                last_dx_sign = dx_sign
                
            if abs(dy) > 2.0:
                dy_sign = 1 if dy > 0 else -1
                if last_dy_sign != 0 and dy_sign != last_dy_sign: y_flips += 1
                last_dy_sign = dy_sign
                
        return dist, x_dist, y_dist, (x_flips + y_flips)

    def get_data(self):
        """Returns a dict with predicted state, intensity, and a raw struggle score."""
        current_time = time.time()
        
        while self.history and current_time - self.history[0][0] > self.long_window:
            self.history.popleft()

        short_cutoff = current_time - self.short_window
        short_history = [p for p in self.history if p[0] >= short_cutoff]

        # Phase 1: Active check
        if len(self.history) > 5:
            long_dist, _, _, _ = self._calc_kinematics(self.history)
            if long_dist > 400:
                self.is_active_reader = True
        
        if not self.history or len(self.history) < 2:
             self.is_active_reader = False
             self.smoothed_intensity = 0.0

        if not self.is_active_reader:
            return {"state": "Inactive", "intensity": 0.0, "score": None}

        # Phase 2: State Prediction (Calculate raw state first)
        raw_state = "Undefined"
        raw_intensity = 0.0
        raw_score = 0.0 
        
        if len(short_history) >= 3:
            short_dist, x_dist, y_dist, short_flips = self._calc_kinematics(short_history)
            speed = short_dist / self.short_window

            if short_dist < 50:
                raw_state = "Undefined"
                raw_intensity = 0.0
                raw_score = 0.0 
            elif short_dist > 300 and short_flips > 8:
                raw_state = "Distracted"
                raw_intensity = min(1.0, (short_flips - 8) / 12.0)
                raw_score = raw_intensity
            else:
                if speed > 400 or y_dist > (x_dist * 1.5):
                    raw_state = "Skimming"
                    raw_intensity = min(1.0, speed / 800.0)
                else:
                    raw_state = "Reading (Focused)"
                    raw_intensity = min(1.0, speed / 300.0)

        # NEW Phase 2.5: Discrete State Smoothing (Majority Vote)
        self.state_buffer.append(raw_state)
        # Find the most common state in the buffer (e.g., last 30 frames)
        smoothed_state = Counter(self.state_buffer).most_common(1)[0][0]

        # Phase 3: Temporal Smoothing (Intensity)
        alpha = 0.05 
        self.smoothed_intensity = (alpha * raw_intensity) + ((1.0 - alpha) * self.smoothed_intensity)

        return {
            "state": smoothed_state,  # <-- Return the smoothed state, not the raw one
            "intensity": self.smoothed_intensity, 
            "score": raw_score if smoothed_state == "Distracted" else 0.0
        }