"""
emotion_wheel.py
================
Emotion detection from MediaPipe face blendshapes,
visualised as Plutchik's Wheel of Emotions using OpenCV.

Public API
----------
    EmotionDetector   – feeds blendshapes, returns smoothed {emotion: score}
    PlutchikWheel     – renders the wheel onto a numpy BGR canvas
    EMOTIONS          – ordered list of the 8 basic emotions
    EMOTION_COLORS    – BGR palette (matches wheel sectors)
"""

import cv2
import numpy as np

# ── Plutchik's 8 basic emotions (clockwise from top) ────────────────────────
EMOTIONS = [
    "Joy", "Trust", "Fear", "Surprise",
    "Sadness", "Disgust", "Anger", "Anticipation",
]

# BGR colour for each emotion sector
EMOTION_COLORS: dict[str, tuple[int, int, int]] = {
    "Joy":          (  0, 215, 255),   # gold
    "Trust":        ( 80, 160,  80),   # green
    "Fear":         ( 40, 200,  40),   # lime
    "Surprise":     (200, 180,  80),   # amber
    "Sadness":      (180,  80,  80),   # steel blue
    "Disgust":      (150,   0, 150),   # purple
    "Anger":        (  0,   0, 210),   # crimson
    "Anticipation": (  0, 140, 255),   # orange
}

# ── Blendshape → emotion contribution weights ────────────────────────────────
# MediaPipe outputs 52 blendshape coefficients (0–1 each).
# We take a weighted sum to score each Plutchik emotion.
_BLENDSHAPE_WEIGHTS: dict[str, dict[str, float]] = {
    "Joy": {
        "mouthSmileLeft":   0.40,
        "mouthSmileRight":  0.40,
        "cheekSquintLeft":  0.10,
        "cheekSquintRight": 0.10,
    },
    "Sadness": {
        "mouthFrownLeft":   0.35,
        "mouthFrownRight":  0.35,
        "browInnerUp":      0.20,
        "eyeSquintLeft":    0.05,
        "eyeSquintRight":   0.05,
    },
    "Anger": {
        "browDownLeft":     0.30,
        "browDownRight":    0.30,
        "noseSneerLeft":    0.15,
        "noseSneerRight":   0.15,
        "mouthPressLeft":   0.05,
        "mouthPressRight":  0.05,
    },
    "Fear": {
        "browInnerUp":      0.25,
        "eyeWideLeft":      0.30,
        "eyeWideRight":     0.30,
        "jawOpen":          0.15,
    },
    "Surprise": {
        "browOuterUpLeft":  0.25,
        "browOuterUpRight": 0.25,
        "eyeWideLeft":      0.20,
        "eyeWideRight":     0.20,
        "jawOpen":          0.10,
    },
    "Disgust": {
        "noseSneerLeft":      0.35,
        "noseSneerRight":     0.35,
        "mouthUpperUpLeft":   0.15,
        "mouthUpperUpRight":  0.15,
    },
    "Trust": {
        "mouthSmileLeft":   0.25,
        "mouthSmileRight":  0.25,
        "eyeSquintLeft":    0.25,
        "eyeSquintRight":   0.25,
    },
    "Anticipation": {
        "browInnerUp":      0.30,
        "eyeWideLeft":      0.15,
        "eyeWideRight":     0.15,
        "mouthSmileLeft":   0.20,
        "mouthSmileRight":  0.20,
    },
}


# ============================================================
# EmotionDetector
# ============================================================
class EmotionDetector:
    """
    Converts MediaPipe face blendshapes to Plutchik emotion scores in [0, 1].

    Parameters
    ----------
    smoothing : float
        EMA coefficient (0 = frozen, 1 = no smoothing).  0.20 gives a
        ~5-frame lag which is enough to suppress single-frame noise.
    """

    def __init__(self, smoothing: float = 0.20):
        self._alpha  = smoothing
        self.scores  = {e: 0.0 for e in EMOTIONS}

    # ── public ──────────────────────────────────────────────────────────────
    def update(self, blendshapes) -> dict[str, float]:
        """
        Parameters
        ----------
        blendshapes
            List of objects with ``.category_name`` (str) and ``.score``
            (float) attributes — exactly what MediaPipe FaceLandmarker returns
            in ``result.face_blendshapes[0]``.
            Pass ``None`` when no face is detected.

        Returns
        -------
        dict mapping each emotion name to a smoothed score in [0, 1].
        """
        if not blendshapes:
            # Decay toward zero so the wheel fades when face is lost
            for e in EMOTIONS:
                self.scores[e] *= (1.0 - self._alpha)
            return self.scores

        bs = {b.category_name: float(b.score) for b in blendshapes}

        for emotion, weights in _BLENDSHAPE_WEIGHTS.items():
            raw = sum(bs.get(name, 0.0) * w for name, w in weights.items())
            self.scores[emotion] = (
                (1.0 - self._alpha) * self.scores[emotion]
                + self._alpha * float(np.clip(raw, 0.0, 1.0))
            )

        return self.scores

    def dominant(self) -> tuple[str, float]:
        """Return (emotion_name, score) for the highest-scoring emotion."""
        e = max(self.scores, key=self.scores.get)
        return e, self.scores[e]

    def reset(self):
        for e in EMOTIONS:
            self.scores[e] = 0.0


# ============================================================
# PlutchikWheel
# ============================================================
class PlutchikWheel:
    """
    Draws an animated Plutchik's Wheel on a numpy BGR canvas.

    Layout
    ------
    8 sectors, 3 concentric rings (mild / moderate / intense).
    Each sector is filled from the hub outward proportional to the score:
      0.00 – 0.33 : mild ring lights up
      0.33 – 0.66 : moderate ring lights up
      0.66 – 1.00 : intense ring lights up

    Parameters
    ----------
    width, height : int
        Size of the returned canvas in pixels.
    """

    _N_RINGS    = 3
    _STEP_ANGLE = 360.0 / len(EMOTIONS)   # 45° per sector

    def __init__(self, width: int = 960, height: int = 540):
        self.w  = width
        self.h  = height
        self.cx = width  // 2
        self.cy = height // 2

        rad              = min(width, height) // 2 - 28
        self.r_inner     = max(int(rad * 0.14), 18)
        self.r_outer     = rad
        span             = self.r_outer - self.r_inner

        # Radii for the 3 ring boundaries
        self._ring_r = [
            self.r_inner + int(span * k / self._N_RINGS)
            for k in range(self._N_RINGS + 1)
        ]

    # ── geometry helpers ────────────────────────────────────────────────────
    def _annulus_pts(self, r0: float, r1: float,
                     a_start: float, a_end: float,
                     steps: int = 48) -> np.ndarray:
        """Polygon approximating an annular sector."""
        a  = np.linspace(np.radians(a_start), np.radians(a_end), steps)
        outer = np.stack([self.cx + r1 * np.cos(a),
                          self.cy + r1 * np.sin(a)], axis=1)
        inner = np.stack([self.cx + r0 * np.cos(a[::-1]),
                          self.cy + r0 * np.sin(a[::-1])], axis=1)
        return np.concatenate([outer, inner], axis=0).astype(np.int32)

    # ── sector renderer ──────────────────────────────────────────────────────
    def _draw_sector(self, canvas: np.ndarray, idx: int, score: float):
        emotion  = EMOTIONS[idx]
        color    = EMOTION_COLORS[emotion]
        s_deg    = idx * self._STEP_ANGLE - 90.0 - self._STEP_ANGLE / 2.0
        e_deg    = s_deg + self._STEP_ANGLE
        r0       = self.r_inner
        r_max    = self.r_outer
        r_fill   = r0 + (r_max - r0) * float(np.clip(score, 0.0, 1.0))

        # ── dim base (always visible at low opacity) ─────────────────────
        dim = tuple(max(int(c * 0.18), 0) for c in color)
        cv2.fillPoly(canvas,
                     [self._annulus_pts(r0, r_max, s_deg, e_deg)], dim)

        # ── bright fill proportional to score ────────────────────────────
        if score > 0.01:
            cv2.fillPoly(canvas,
                         [self._annulus_pts(r0, r_fill, s_deg, e_deg)], color)

        # ── ring tier separators (thin dark arcs) ────────────────────────
        for r in self._ring_r[1:-1]:
            # Draw as thin annulus so it works without cv2.polylines thickness issues
            pts = self._annulus_pts(r - 1, r + 1, s_deg, e_deg, steps=36)
            cv2.fillPoly(canvas, [pts], (12, 12, 12))

        # ── spokes ───────────────────────────────────────────────────────
        for deg in (s_deg, e_deg):
            rad = np.radians(deg)
            p0  = (int(self.cx + r0    * np.cos(rad)),
                   int(self.cy + r0    * np.sin(rad)))
            p1  = (int(self.cx + r_max * np.cos(rad)),
                   int(self.cy + r_max * np.sin(rad)))
            cv2.line(canvas, p0, p1, (12, 12, 12), 1)

        # ── emotion label (mid-radius) ───────────────────────────────────
        mid_rad = np.radians((s_deg + e_deg) / 2.0)
        r_lbl   = r0 + (r_max - r0) * 0.60
        lx = int(self.cx + r_lbl * np.cos(mid_rad))
        ly = int(self.cy + r_lbl * np.sin(mid_rad))
        fs = 0.42
        tw, th = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0]
        cv2.putText(canvas, emotion,
                    (lx - tw // 2, ly + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (230, 230, 230), 1, cv2.LINE_AA)

        # ── score badge (inner ring) ─────────────────────────────────────
        badge = f"{score:.2f}"
        fs2 = 0.30
        bw, bh = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, fs2, 1)[0]
        r_badge = r0 + (r_max - r0) * 0.24
        bx = int(self.cx + r_badge * np.cos(mid_rad)) - bw // 2
        by = int(self.cy + r_badge * np.sin(mid_rad)) + bh // 2
        cv2.putText(canvas, badge, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, fs2,
                    (170, 170, 170), 1, cv2.LINE_AA)

    # ── public render ────────────────────────────────────────────────────────
    def render(self, scores: dict[str, float]) -> np.ndarray:
        """
        Parameters
        ----------
        scores : dict  { emotion_name : float [0, 1] }

        Returns
        -------
        numpy array, shape (height, width, 3), dtype uint8, BGR.
        """
        canvas = np.full((self.h, self.w, 3), 18, dtype=np.uint8)

        # Draw all 8 sectors
        for idx, emotion in enumerate(EMOTIONS):
            self._draw_sector(canvas, idx, scores.get(emotion, 0.0))

        # Outer rim + inner hub circles
        cv2.circle(canvas, (self.cx, self.cy), self.r_outer, (70, 70, 70), 1)
        cv2.circle(canvas, (self.cx, self.cy), self.r_inner, (70, 70, 70), 1)

        # Centre text
        for txt, dy in [("Plutchik", -6), ("Wheel", 10)]:
            tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)[0][0]
            cv2.putText(canvas, txt,
                        (self.cx - tw // 2, self.cy + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                        (110, 110, 110), 1, cv2.LINE_AA)

        # Panel title
        cv2.putText(canvas, "Emotion Detector",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    (180, 180, 180), 1, cv2.LINE_AA)

        # Ring intensity legend (top-right corner)
        legend = [("mild", 0), ("moderate", 1), ("intense", 2)]
        for label, ring_idx in legend:
            r_mid = (self._ring_r[ring_idx] + self._ring_r[ring_idx + 1]) // 2
            cv2.putText(canvas, label,
                        (self.w - 72, self.cy - r_mid + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        (100, 100, 100), 1, cv2.LINE_AA)

        # Dominant emotion banner (bottom of panel)
        dom = max(scores, key=scores.get) if scores else "—"
        dom_score = scores.get(dom, 0.0)
        if dom_score > 0.05:
            bar = f"Dominant:  {dom}  ({dom_score:.2f})"
            cv2.putText(canvas, bar,
                        (10, self.h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        EMOTION_COLORS.get(dom, (200, 200, 200)),
                        1, cv2.LINE_AA)

        return canvas