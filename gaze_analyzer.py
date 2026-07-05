import time
import math
from collections import deque

class GazeReadingAnalyzer:
    def __init__(self, window_size=4.0):
        """
        Analyzes eye gaze coordinates over a sliding window to detect 
        cognitive struggle and classify specific reading states.
        """
        self.window_size = window_size
        self.history = deque()

    def process_point(self, x, y, on_paper=True):
        """
        Feed the smoothed Kalman Gaze (x,y) coordinates here every frame.
        `on_paper` should be a boolean indicating if the (x,y) coordinate intersects the PDF bounding box.
        
        Returns a dictionary: {'score': float, 'state': str}
        """
        current_time = time.time()
        self.history.append((current_time, x, y, on_paper))
        
        # Slide window to drop old data
        while self.history and current_time - self.history[0][0] > self.window_size:
            self.history.popleft()
            
        return self._analyze_window()

    def _analyze_window(self):
        # Default return if we don't have enough data yet
        if len(self.history) < 10:
            return {"score": 0.0, "state": "Initializing"}

        regressions = 0
        long_fixations = 0
        
        # --- Parameters ---
        REGRESSION_X_THRESHOLD = -30  # Jumped back 30+ pixels
        LINE_CHANGE_Y_THRESHOLD = 40  # Moved down a line
        FIXATION_RADIUS = 25          # Pixels
        LONG_FIXATION_SECONDS = 0.85  # Require longer fixations before penalizing
        
        # 1. Count Regressions
        for i in range(1, len(self.history)):
            dx = self.history[i][1] - self.history[i-1][1]
            dy = self.history[i][2] - self.history[i-1][2]
            
            if dx < REGRESSION_X_THRESHOLD and abs(dy) < LINE_CHANGE_Y_THRESHOLD:
                regressions += 1

        # 2. Detect Prolonged Fixations
        fixation_start_idx = 0
        while fixation_start_idx < len(self.history):
            start_t, start_x, start_y, _ = self.history[fixation_start_idx]
            
            end_idx = fixation_start_idx
            while end_idx < len(self.history):
                curr_t, curr_x, curr_y, _ = self.history[end_idx]
                dist = math.sqrt((curr_x - start_x)**2 + (curr_y - start_y)**2)
                
                if dist > FIXATION_RADIUS:
                    break 
                end_idx += 1
                
            duration = self.history[end_idx-1][0] - start_t
            if duration > LONG_FIXATION_SECONDS: # Require a longer sustained fixation
                long_fixations += 1
                
            fixation_start_idx = end_idx + 1

        # 3. Calculate Struggle Score
        regression_penalty = min(1.0, regressions / 4.0)
        fixation_penalty = min(1.0, long_fixations / 3.0)
        struggle_score = max(0.0, min(1.0, (regression_penalty * 0.40) + (fixation_penalty * 0.60)))
        
        # 4. Determine Reading State
        current_state = self._determine_state(struggle_score, long_fixations, regressions)

        return {
            "score": struggle_score,
            "state": current_state
        }

    def _determine_state(self, struggle_score, fixations, regressions):
        """
        Applies spatial heuristics to classify what the user is actually doing.
        """
        on_paper_count = sum(1 for item in self.history if item[3] is True)
        on_paper_ratio = on_paper_count / len(self.history)
        
        first_y = self.history[0][2]
        last_y = self.history[-1][2]
        dy_total = last_y - first_y 
        
        xs = [h[1] for h in self.history]
        ys = [h[2] for h in self.history]
        spread_x = max(xs) - min(xs)
        spread_y = max(ys) - min(ys)
        
        # NEW: Calculate where on the screen the user is primarily looking
        avg_y = sum(ys) / len(ys) if ys else 0
        
        # --- State Decision Tree ---
        if on_paper_ratio < 0.50:
            return "Thinking (Off-Text)"
            
        if struggle_score > 0.70 and (fixations > 1 or regressions > 0):
            return "Struggling / Stuck"
            
        if spread_y > 400 and spread_x > 800:
            return "Distracted"
            

            
        if dy_total < -100:
            return "Scanning (Upwards)"
            
        if spread_x < 100 and spread_y < 100:
            # NEW: If looking at the bottom, downgrade to regular reading
            if avg_y > 750: 
                return "Reading (Focus  ed)"
            return "Deep Focus"
            
        return "Reading (Focused)"