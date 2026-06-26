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
            if duration > 0.500: # 500 milliseconds
                long_fixations += 1
                
            fixation_start_idx = end_idx + 1

        # 3. Calculate Struggle Score
        regression_penalty = min(1.0, regressions / 4.0)       
        fixation_penalty = min(1.0, long_fixations / 2.0)      
        struggle_score = max(0.0, min(1.0, (regression_penalty * 0.40) + (fixation_penalty * 0.60)))
        
        # 4. Determine Reading State
        current_state = self._determine_state(struggle_score, long_fixations)

        return {
            "score": struggle_score,
            "state": current_state
        }

    def _determine_state(self, struggle_score, fixations):
        """
        Applies spatial heuristics to classify what the user is actually doing.
        """
        # Calculate ratio of time spent looking at the actual text
        on_paper_count = sum(1 for item in self.history if item[3] is True)
        on_paper_ratio = on_paper_count / len(self.history)
        
        # Calculate vertical progress (Positive = moving down the page)
        first_y = self.history[0][2]
        last_y = self.history[-1][2]
        dy_total = last_y - first_y 
        
        # Calculate spatial bounding box to detect erratic vs focused movement
        xs = [h[1] for h in self.history]
        ys = [h[2] for h in self.history]
        spread_x = max(xs) - min(xs)
        spread_y = max(ys) - min(ys)
        
        # --- State Decision Tree ---
        
        # 1. Thinking / Zoning Out (Gaze is predominantly off the paper)
        if on_paper_ratio < 0.50:
            return "Thinking (Off-Text)"
            
        # 2. Struggling (Score is high, overriding normal reading mechanics)
        if struggle_score > 0.60:
            return "Struggling / Stuck"
            
        # 3. Distracted (Massive, erratic jumps across the screen)
        if spread_y > 400 and spread_x > 800:
            return "Distracted"
            
        # 4. Skimming (Fast downward movement, few heavy fixations)
        if dy_total > 150 and fixations <= 1 and spread_x < 600:
            return "Skimming"
            
        # 5. Scanning / Searching (Moving fast *upwards* to re-read earlier context)
        if dy_total < -100:
            return "Scanning (Upwards)"
            
        # 6. Deep Focus (Staring at a very tight area, e.g., a diagram, without struggling)
        if spread_x < 100 and spread_y < 100:
            return "Deep Focus"
            
        # 7. Default behavior
        return "Reading (Focused)"