import math
from collections import deque

class ScrollKinematicsAnalyzer:
    def __init__(self, prr_top=0.25, prr_bottom=0.45):
        self.prr_top = prr_top       # 25% down the screen
        self.prr_bottom = prr_bottom # 45% down the screen
        self.scroll_history = deque(maxlen=30)
        self.last_scroll_y = 0
        self.is_skimming = False
        
    def analyze(self, current_scroll_y, dt):
        velocity = abs(current_scroll_y - self.last_scroll_y) / (dt + 1e-6)
        self.scroll_history.append(velocity)
        self.last_scroll_y = current_scroll_y
        
        # Determine if user is skimming (High velocity over last 30 frames)
        avg_velocity = sum(self.scroll_history) / len(self.scroll_history)
        self.is_skimming = avg_velocity > 50.0 
        
        return self.is_skimming

    def get_prr_y(self, screen_h):
        # Returns the center pixel Y-coordinate of the Preferred Reading Region
        return screen_h * ((self.prr_top + self.prr_bottom) / 2)