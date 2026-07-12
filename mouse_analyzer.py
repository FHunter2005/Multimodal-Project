import time
import math
from collections import deque, Counter 
from pynput import mouse

class MouseReadingAnalyzer:
    def __init__(self, short_window=4.0, long_window=10.0, state_buffer_size=30):
        self.short_window = short_window
        self.long_window = long_window
        self.history = deque()
        
        self.is_active_reader = False
        self.smoothed_intensity = 0.0
        self._listener = None
        
        self.state_buffer = deque(maxlen=state_buffer_size) 

    def start(self):
        self._listener = mouse.Listener(on_move=self._on_move)
        self._listener.start()
        print("[INFO] Mouse Modality Started (Kinematic State Detection).")

    def stop(self):
        if self._listener:
            self._listener.stop()

    def detect_circling_gesture(self, time_window=1.5):
        """
        Evaluates recent mouse history to detect a circular/looping gesture.
        Returns the center (cx, cy) of the circle if detected, else None.
        """
        current_time = time.time()
        # Filter points within the target time window
        recent_points = [p for p in self.history if current_time - p[0] <= time_window]
        
        if len(recent_points) < 15: # Not enough data points to form a circle
            return None
            
        xs = [p[1] for p in recent_points]
        ys = [p[2] for p in recent_points]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # 1. Size constraint: The gesture must be large enough to be deliberate
        if width < 60 or height < 60:
            return None
            
        # 2. Aspect Ratio: A circle/loop should be somewhat proportional, 
        # avoiding long horizontal reading sweeps
        aspect_ratio = width / height if height > 0 else 100
        if not (0.3 < aspect_ratio < 3.0):
            return None
            
        # 3. Calculate Path Length vs Net Displacement
        path_length = 0.0
        for i in range(1, len(recent_points)):
            dx = recent_points[i][1] - recent_points[i-1][1]
            dy = recent_points[i][2] - recent_points[i-1][2]
            path_length += math.sqrt(dx**2 + dy**2)
            
        start_pt = recent_points[0]
        end_pt = recent_points[-1]
        net_displacement = math.sqrt((end_pt[1] - start_pt[1])**2 + (end_pt[2] - start_pt[2])**2)
        
        # 4. Circle Logic: The path length must be significantly larger than 
        # the net displacement (meaning it looped back on itself)
        if path_length > 2.5 * max(width, height) and net_displacement < (path_length * 0.4):
            # Calculate the center of the bounding box
            center_x = min_x + (width / 2)
            center_y = min_y + (height / 2)
            return (int(center_x), int(center_y))
            
        return None

    def _on_move(self, x, y):
        self.history.append((time.time(), x, y))

    def _calc_kinematics(self, points):
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

        if len(self.history) > 5:
            long_dist, _, _, _ = self._calc_kinematics(self.history)
            if long_dist > 400:
                self.is_active_reader = True
        
        if not self.history or len(self.history) < 2:
             self.is_active_reader = False
             self.smoothed_intensity = 0.0

        if not self.is_active_reader:
            return {"state": "Inactive", "intensity": 0.0, "score": None}

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

        self.state_buffer.append(raw_state)
        
        smoothed_state = Counter(self.state_buffer).most_common(1)[0][0]

        alpha = 0.05 
        self.smoothed_intensity = (alpha * raw_intensity) + ((1.0 - alpha) * self.smoothed_intensity)

        return {
            "state": smoothed_state,  
            "intensity": self.smoothed_intensity, 
            "score": raw_score if smoothed_state == "Distracted" else 0.0
        }