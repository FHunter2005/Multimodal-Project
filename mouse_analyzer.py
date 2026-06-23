import time
import math
from collections import deque
from pynput import mouse

class MouseReadingAnalyzer:
    def __init__(self, short_window=4.0, long_window=10.0):
        """
        Calculates a stable, real-time 'Struggle Score' [0.0 to 1.0].
        Outputs 'None' if the user is not actively using the mouse.
        """
        self.short_window = short_window
        self.long_window = long_window
        self.history = deque()
        
        self.is_active_reader = False
        self.smoothed_score = 0.0  # Added for UI stability
        self._listener = None

    def start(self):
        self._listener = mouse.Listener(on_move=self._on_move)
        self._listener.start()
        print("[INFO] Mouse Modality Started (Kinematic Chaos Detection).")

    def stop(self):
        if self._listener:
            self._listener.stop()

    def _on_move(self, x, y):
        self.history.append((time.time(), x, y))

    def _calc_kinematics(self, points):
        """Calculates distance and counts how many times the mouse reverses direction."""
        if len(points) < 2:
            return 0.0, 0
            
        dist = 0.0
        x_flips = 0
        y_flips = 0
        
        last_dx_sign = 0
        last_dy_sign = 0

        for i in range(1, len(points)):
            dx = points[i][1] - points[i-1][1]
            dy = points[i][2] - points[i-1][2]
            dist += math.sqrt(dx**2 + dy**2)
            
            # Count X-axis reversals (ignore tiny jitter < 2px)
            if abs(dx) > 2.0:
                dx_sign = 1 if dx > 0 else -1
                if last_dx_sign != 0 and dx_sign != last_dx_sign:
                    x_flips += 1
                last_dx_sign = dx_sign
                
            # Count Y-axis reversals
            if abs(dy) > 2.0:
                dy_sign = 1 if dy > 0 else -1
                if last_dy_sign != 0 and dy_sign != last_dy_sign:
                    y_flips += 1
                last_dy_sign = dy_sign
                
        total_flips = x_flips + y_flips
        return dist, total_flips

    def get_score(self):
        """Polled by the main application loop. Returns smoothed float [0, 1] or None."""
        current_time = time.time()
        
        # 1. Slide the long window
        while self.history and current_time - self.history[0][0] > self.long_window:
            self.history.popleft()

        short_cutoff = current_time - self.short_window
        short_history = [p for p in self.history if p[0] >= short_cutoff]

        # --- PHASE 1: User Profiling ---
        if len(self.history) > 5:
            long_dist, _ = self._calc_kinematics(self.history)
            if long_dist > 400:
                self.is_active_reader = True
        
        # Drop active status if they let go of the mouse
        if not self.history or len(self.history) < 2:
             self.is_active_reader = False
             self.smoothed_score = 0.0

        if not self.is_active_reader:
            return None

        # --- PHASE 2: Immediate Behavior Analysis ---
        raw_score = 0.0
        
        if len(short_history) < 3:
            raw_score = 1.0  # Completely stopped
        else:
            short_dist, short_flips = self._calc_kinematics(short_history)

            # Anomaly 1: Hovering/Stuck (Low distance)
            if short_dist < 50:
                raw_score = 1.0

            # Anomaly 2: Chaotic Frustration (High distance + High Reversals)
            # Normal reading creates ~2 to 5 flips. 
            # Shaking/scribbling quickly generates 10+ flips.
            elif short_dist > 300 and short_flips > 8:
                # Scale intensity based on how chaotic it is (caps at ~20 flips)
                raw_score = min(1.0, (short_flips - 8) / 12.0)

            # Normal Behavior: Engaged Tracing
            # High distance, but low flips (tracing straight lines in ANY direction)
            else:
                raw_score = 0.0

        # --- PHASE 3: Temporal Smoothing (EMA) ---
        # This prevents the score from flickering or jumping instantly.
        # It glides smoothly toward the raw_score.
        alpha = 0.05 # Smoothing factor. Lower = smoother but slightly delayed.
        self.smoothed_score = (alpha * raw_score) + ((1.0 - alpha) * self.smoothed_score)

        return self.smoothed_score