import time
import math
from collections import deque

class GazeReadingAnalyzer:
    def __init__(self, window_size=4.0):
        """
        Analyzes eye gaze coordinates over a sliding window to detect 
        cognitive struggle during reading (fixations and regressions).
        """
        self.window_size = window_size
        self.history = deque()

    def process_point(self, x, y):
        """
        Feed the smoothed Kalman Gaze (x,y) coordinates here every frame.
        Returns a Gaze Struggle Score [0.0, 1.0].
        """
        current_time = time.time()
        self.history.append((current_time, x, y))
        
        # Slide window
        while self.history and current_time - self.history[0][0] > self.window_size:
            self.history.popleft()
            
        return self._calculate_score()

    def _calculate_score(self):
        if len(self.history) < 10:
            return 0.0

        regressions = 0
        long_fixations = 0
        
        # Parameters for reading heuristics (assuming 1920x1080 screen)
        # Adjust these based on your specific UI text size
        REGRESSION_X_THRESHOLD = -30  # Jumped back 30+ pixels
        LINE_CHANGE_Y_THRESHOLD = 40  # Moved down a line (ignore these regressions)
        FIXATION_RADIUS = 25          # Pixels
        
        # 1. Count Regressions (Backward Jumps)
        for i in range(1, len(self.history)):
            dx = self.history[i][1] - self.history[i-1][1]
            dy = self.history[i][2] - self.history[i-1][2]
            
            # If they moved left, but didn't drop down a line, it's a regression
            if dx < REGRESSION_X_THRESHOLD and abs(dy) < LINE_CHANGE_Y_THRESHOLD:
                regressions += 1

        # 2. Detect Prolonged Fixations (Stuck on a word)
        # We check if the gaze stayed within a small radius for > 500ms
        fixation_start_idx = 0
        while fixation_start_idx < len(self.history):
            start_t, start_x, start_y = self.history[fixation_start_idx]
            
            end_idx = fixation_start_idx
            while end_idx < len(self.history):
                curr_t, curr_x, curr_y = self.history[end_idx]
                dist = math.sqrt((curr_x - start_x)**2 + (curr_y - start_y)**2)
                
                if dist > FIXATION_RADIUS:
                    break # Gaze moved out of the fixation zone
                end_idx += 1
                
            duration = self.history[end_idx-1][0] - start_t
            if duration > 0.500: # 500 milliseconds
                long_fixations += 1
                
            # Jump ahead to avoid double-counting the same fixation
            fixation_start_idx = end_idx + 1

        # 3. Calculate Final Struggle Score
        # Normal reading might have 1 regression and 0 long fixations per 4 seconds.
        regression_penalty = min(1.0, regressions / 4.0)       # Caps at 4 regressions
        fixation_penalty = min(1.0, long_fixations / 2.0)      # Caps at 2 long fixations
        
        # Weighted combination (Fixations are stronger indicators of being totally stuck)
        struggle_score = (regression_penalty * 0.40) + (fixation_penalty * 0.60)
        
        return max(0.0, min(1.0, struggle_score))