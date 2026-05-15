import time
import collections

class FixationDetector:
    """
    Detects fixations (stable gaze on a point) vs saccades (fast movements).

    How it works:
    - Maintains a rolling window of recent gaze positions
    - If gaze stays within DISPERSION_THRESHOLD for MIN_DURATION seconds
      → it's a fixation
    - If gaze moves faster than the threshold → it's a saccade

    For the reading helper:
    - Fixation on a paragraph zone = reading that paragraph
    - Long fixation on same spot = possibly confused
    - Many short fixations = scanning/searching
    - Re-fixation on already-read zone = re-reading (confusion signal)
    """

    # Gaze must stay within this radius (in normalized 0-1 screen coords)
    # 0.05 = 5% of screen width/height — tune this for sensitivity
    DISPERSION_THRESHOLD = 0.06

    # Gaze must be stable for at least this long to count as a fixation
    MIN_DURATION = 0.15  # seconds

    # How long a fixation can last before flagging as "stuck" (confusion)
    CONFUSION_DURATION = 3.0  # seconds

    def __init__(self):
        self.window      = collections.deque()  # (timestamp, x, y)
        self.in_fixation = False
        self.fixation_start = None
        self.fixation_x  = None
        self.fixation_y  = None

        # History of completed fixations for pattern analysis
        self.fixation_history = []  # list of dicts

    def update(self, x, y):
        """
        Feed a new gaze point (normalized 0-1).
        Returns a dict with current state.
        """
        now = time.time()
        self.window.append((now, x, y))

        # Drop points older than CONFUSION_DURATION (max window we need)
        while self.window and now - self.window[0][0] > self.CONFUSION_DURATION:
            self.window.popleft()

        # Need at least 2 points
        if len(self.window) < 2:
            return self._state(False, False, x, y, 0.0)

        # Calculate dispersion: max distance between any two points in window
        xs = [p[1] for p in self.window]
        ys = [p[2] for p in self.window]
        dispersion = max(
            ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
            for x1, y1 in zip(xs, ys)
            for x2, y2 in zip(xs, ys)
        )

        # But only look at recent window for fixation detection
        recent_cutoff = now - self.MIN_DURATION
        recent = [(t, rx, ry) for t, rx, ry in self.window if t >= recent_cutoff]

        if len(recent) < 2:
            return self._state(False, False, x, y, 0.0)

        recent_xs = [p[1] for p in recent]
        recent_ys = [p[2] for p in recent]
        recent_dispersion = (
            (max(recent_xs) - min(recent_xs))**2 +
            (max(recent_ys) - min(recent_ys))**2
        ) ** 0.5

        is_fixation = recent_dispersion < self.DISPERSION_THRESHOLD

        if is_fixation:
            if not self.in_fixation:
                # Fixation just started
                self.in_fixation    = True
                self.fixation_start = now
                self.fixation_x     = sum(recent_xs) / len(recent_xs)
                self.fixation_y     = sum(recent_ys) / len(recent_ys)

            duration = now - self.fixation_start
            is_confused = duration >= self.CONFUSION_DURATION

            return self._state(True, is_confused,
                               self.fixation_x, self.fixation_y, duration)
        else:
            if self.in_fixation:
                # Fixation just ended — record it
                duration = now - self.fixation_start
                self.fixation_history.append({
                    'x':        self.fixation_x,
                    'y':        self.fixation_y,
                    'duration': duration,
                    'end_time': now,
                })
                self.in_fixation = False

            return self._state(False, False, x, y, 0.0)

    def _state(self, is_fixation, is_confused, fx, fy, duration):
        return {
            'is_fixation':  is_fixation,
            'is_confused':  is_confused,   # fixated too long on same spot
            'fixation_x':   fx,
            'fixation_y':   fy,
            'duration':     duration,
            'history':      self.fixation_history,
        }

    def get_blink_adjusted(self):
        """
        Returns recent fixation history for pattern analysis.
        Useful for detecting re-reading behavior.
        """
        now = time.time()
        recent = [f for f in self.fixation_history
                  if now - f['end_time'] < 30.0]  # last 30 seconds
        return recent

    def reset(self):
        self.window.clear()
        self.in_fixation    = False
        self.fixation_start = None
        self.fixation_x     = None
        self.fixation_y     = None
        self.fixation_history.clear()