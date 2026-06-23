import time
import math
from collections import deque
from pynput import mouse

class MouseReadingAnalyzer:
    def __init__(self, window_size=3.0):
        """
        Runs a background listener for mouse movements and calculates
        a real-time reading concentration score [0, 1].
        """
        self.window_size = window_size
        self.history = deque()
        self.current_score = 0.0
        self._listener = None

    def start(self):
        """Starts the background pynput listener."""
        self._listener = mouse.Listener(on_move=self._on_move)
        self._listener.start()
        print("[INFO] Mouse tracking modality started.")

    def stop(self):
        """Stops the listener."""
        if self._listener:
            self._listener.stop()

    def _on_move(self, x, y):
        """Callback for pynput. Process point and update sliding window."""
        current_time = time.time()
        self.history.append((current_time, x, y))
        
        # Slide window: remove old data
        while self.history and current_time - self.history[0][0] > self.window_size:
            self.history.popleft()
            
        self.current_score = self._calculate_score()

    def _calculate_score(self):
        """Heuristics to determine if mouse movement matches reading patterns."""
        if len(self.history) < 5:
            return 0.0 
            
        total_dx_abs = 0.0
        total_dy_abs = 0.0
        total_distance = 0.0
        
        for i in range(1, len(self.history)):
            dx = self.history[i][1] - self.history[i-1][1]
            dy = self.history[i][2] - self.history[i-1][2]
            total_dx_abs += abs(dx)
            total_dy_abs += abs(dy)
            total_distance += math.sqrt(dx**2 + dy**2)
            
        # Metric A: Horizontal Bias (Reading sweeps horizontally)
        total_movement = total_dx_abs + total_dy_abs
        horizontal_bias = 0.0 if total_movement == 0 else total_dx_abs / total_movement
            
        # Metric B: Fluency (Targeting 100-1500 pixels per 3 seconds)
        if 100 < total_distance < 1500:
            activity_score = 1.0
        else:
            deviation = abs(total_distance - 800)
            activity_score = max(0.0, 1.0 - (deviation / 1000.0))
            
        # Weighted combination: 70% sweeping, 30% speed
        raw_concentration = (horizontal_bias * 0.70) + (activity_score * 0.30)
        return max(0.0, min(1.0, raw_concentration))

    def get_score(self):
        """Thread-safe getter for the main application loop."""
        return self.current_score