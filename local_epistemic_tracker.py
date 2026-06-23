"""
LocalEpistemicTracker v2 — Multi-modal Epistemic State Detector
================================================================
Priority: Confusion and Frustration accuracy above all else.

SIGNAL ARCHITECTURE
===================
Each state is scored from N independent sub-signals, each normalised
0→1, then combined with calibrated weights. Sub-signals come from
three source layers:

  Layer 1 — Blendshape AUs (MediaPipe facial action units)
  Layer 2 — Head Pose     (yaw, pitch, roll from solvePnP)
  Layer 3 — Temporal      (trends, velocities, event counts derived
                           from the rolling buffer)

The API accepts both layers per frame:
  tracker.update(blendshapes, head_pose=(yaw, pitch, roll, face_scale))

head_pose is optional — tracker degrades gracefully to AU-only mode.

─────────────────────────────────────────────────────────────────────
CONFUSION SIGNALS  (7 sub-signals)
─────────────────────────────────────────────────────────────────────
  [AU]   inner_brow_raise  — browInnerUp high (AU1, oblique pull)
  [AU]   brow_furrow       — browDown moderate (AU4, concentrating)
  [AU]   brow_squint_combo — browInnerUp AND squint together
                             (classic "?" face, distinct from frustration)
  [AU]   brow_asymmetry    — |browDownLeft - browDownRight| sustained
                             (unilateral confusion furrow)
  [POSE] head_tilt         — |roll| > 8° sustained (tilting = "huh?")
  [POSE] lean_in           — face_scale positive trend (approaching screen)
  [TEMP] brow_flicker      — std(browInnerUp) high across window
                             (repeated micro-confusion pulses)

─────────────────────────────────────────────────────────────────────
FRUSTRATION SIGNALS  (7 sub-signals)
─────────────────────────────────────────────────────────────────────
  [AU]   hard_furrow       — browDown high (AU4 sustained, stronger than confusion)
  [AU]   lip_press         — mouthPress (AU28, lips compressed inward)
  [AU]   nose_sneer        — noseSneer (AU9, wrinkle above nostril)
  [AU]   lip_frown         — mouthFrown (AU15, corner pull-down)
  [AU]   jaw_tension       — mouthClose high + jawOpen near zero
                             (clenching without opening)
  [POSE] head_agitation    — std(yaw) + std(pitch) high (restless movement)
  [TEMP] escalation        — positive trend in frustration proxy over long window

─────────────────────────────────────────────────────────────────────
BOREDOM SIGNALS  (6 sub-signals)
─────────────────────────────────────────────────────────────────────
  [AU]   eyes_down         — eyeLookDownLeft/Right mean high
                             (gaze cast downward, not at screen)
  [AU]   face_slack        — overall AU variance very low (blank, relaxed face)
  [AU]   mouth_slack       — mouthClose low + jawOpen near zero + no press/frown
                             (jaw just hanging loose, not clenched)
  [POSE] far_from_screen   — face_scale small + negative trend
                             (leaning back, physically distant)
  [POSE] relaxed_posture   — low head movement variance + upright/neutral pitch
                             (not drooping, just slumped back calmly)
  [POSE] lean_back         — face_scale negative trend (actively moving away)

─────────────────────────────────────────────────────────────────────
CONCENTRATION SIGNALS  (5 sub-signals)
─────────────────────────────────────────────────────────────────────
  [AU]   mild_furrow       — browDown 0.04–0.16 (light focus, not distress)
  [AU]   mouth_still       — low mouth AU variance (engaged, not talking)
  [AU]   eyes_open         — low blink rate
  [POSE] head_stable       — low std(yaw/pitch) — not fidgeting
  [POSE] neutral_pitch     — small |pitch| — upright, alert posture

Cross-suppression matrix applied after raw scoring:
  Frustration ←→ Boredom  (strong mutual inhibition)
  Confusion   → reduces Concentration
  Frustration → reduces Boredom
"""

from collections import deque
from typing import Optional, Tuple, Any
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Temporal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lintrend(arr: np.ndarray) -> float:
    """Least-squares slope (normalised by length). +ve = rising."""
    n = len(arr)
    if n < 4:
        return 0.0
    x = np.arange(n, dtype=np.float32) - (n - 1) / 2.0
    denom = float((x * x).sum())
    if denom < 1e-9:
        return 0.0
    return float((x * (arr - arr.mean())).sum() / denom)


def _count_peaks(arr: np.ndarray, threshold: float, min_gap: int = 5) -> int:
    """Count number of times arr crosses above threshold (with min_gap debounce)."""
    above = arr > threshold
    peaks, last = 0, -min_gap
    for i, a in enumerate(above):
        if a and (i - last) >= min_gap:
            peaks += 1
            last = i
    return peaks


# ─────────────────────────────────────────────────────────────────────────────
# Blendshape keys we care about
# ─────────────────────────────────────────────────────────────────────────────

_BS_KEYS = [
    "browDownLeft",    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "eyeBlinkLeft",    "eyeBlinkRight",
    "eyeSquintLeft",   "eyeSquintRight",
    "eyeWideLeft",     "eyeWideRight",
    "jawOpen",         "mouthClose",
    "mouthFrownLeft",  "mouthFrownRight",
    "mouthPressLeft",  "mouthPressRight",
    "mouthSmileLeft",  "mouthSmileRight",
    "mouthDimpleLeft", "mouthDimpleRight",
    "noseSneerLeft",   "noseSneerRight",
    "cheekSquintLeft", "cheekSquintRight",
    "cheekPuff",
    # Gaze direction — present in MediaPipe face_landmarker blendshapes
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookUpLeft",   "eyeLookUpRight",
]

# Pose keys appended after blendshapes in the feature vector
_POSE_KEYS = ["yaw", "pitch", "roll", "face_scale"]
_ALL_KEYS  = _BS_KEYS + _POSE_KEYS
_KEY_IDX   = {k: i for i, k in enumerate(_ALL_KEYS)}


def _extract(blendshapes, head_pose: Optional[Tuple]) -> Optional[np.ndarray]:
    """
    Build a flat float32 feature vector from one frame's data.
    Returns None if no face detected.
    head_pose: (yaw, pitch, roll, face_scale) or None
    """
    if not blendshapes:
        return None
    bs = {b.category_name: float(b.score) for b in blendshapes}
    vec = np.array([bs.get(k, 0.0) for k in _BS_KEYS], dtype=np.float32)

    if head_pose is not None:
        yaw, pitch, roll, fscale = head_pose
        pose_vec = np.array([
            float(yaw)    / 45.0,   # normalise: ±45° → ±1
            float(pitch)  / 30.0,
            float(roll)   / 45.0,
            float(fscale),          # raw face scale (inter-eye / face-width ratio)
        ], dtype=np.float32)
    else:
        pose_vec = np.zeros(len(_POSE_KEYS), dtype=np.float32)

    return np.concatenate([vec, pose_vec])


def _i(key: str) -> int:
    return _KEY_IDX[key]


# ─────────────────────────────────────────────────────────────────────────────
# Sub-signal library  (pure functions, takes mu / sigma / trend arrays)
# ─────────────────────────────────────────────────────────────────────────────

def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


class _Stats:
    """Thin wrapper around pre-computed window statistics for ergonomic access."""
    def __init__(self, arr: np.ndarray):
        self.arr   = arr                          # (n_valid, n_features)
        self.mu    = arr.mean(axis=0)
        self.sigma = arr.std(axis=0)

    def m(self, key: str) -> float:
        return float(self.mu[_i(key)])

    def s(self, key: str) -> float:
        return float(self.sigma[_i(key)])

    def t(self, key: str) -> float:
        return _lintrend(self.arr[:, _i(key)])

    def col(self, key: str) -> np.ndarray:
        return self.arr[:, _i(key)]

    def avg_m(self, *keys) -> float:
        return float(np.mean([self.m(k) for k in keys]))

    def avg_s(self, *keys) -> float:
        return float(np.mean([self.s(k) for k in keys]))

    def avg_t(self, *keys) -> float:
        return float(np.mean([self.t(k) for k in keys]))


# ─────────────────────────────────────────────────────────────────────────────
# Main tracker
# ─────────────────────────────────────────────────────────────────────────────

class LocalEpistemicTracker:
    """
    Detects Boredom / Confusion / Frustration / Concentration.

    Parameters
    ----------
    window_frames : int   Rolling window length (default 90 ≈ 3 s @30fps)
    smooth_alpha  : float EMA weight for UI animation (0.05–0.20)
    min_frames    : int   Minimum valid frames before scoring
    fps           : int   Expected FPS (used for time labels only)
    """

    STATES = ["Concentration", "Confusion", "Boredom", "Frustration"]

    _COLORS = {
        "Concentration": (200, 175,  50),
        "Confusion":     ( 50, 170, 255),
        "Boredom":       (120, 115, 195),
        "Frustration":   ( 40,  50, 215),
    }

    def __init__(
        self,
        window_frames: int   = 90,
        smooth_alpha:  float = 0.10,
        min_frames:    int   = 20,
        fps:           int   = 30,
    ):
        self.window_frames = window_frames
        self.smooth_alpha  = smooth_alpha
        self.min_frames    = min_frames
        self.fps           = fps

        self._buf: deque = deque(maxlen=window_frames)   # np.ndarray or None
        self._has_pose   = False                         # set on first pose frame

        self.target_scores = {s: 0.0 for s in self.STATES}
        self.smooth_scores = {s: 0.0 for s in self.STATES}

        # Sub-signal scores for the debug panel (Confusion + Frustration)
        self.sub_conf = {}   # name → float
        self.sub_frus = {}

        _hl = 150
        self._history = {s: deque([0.0] * _hl, maxlen=_hl) for s in self.STATES}

        self.dominant_state = "—"
        self.dominant_conf  = 0.0

        # Cached active signal names for the HUD
        self._active_conf_signals: list = []
        self._active_frus_signals: list = []

    # ── public API ────────────────────────────────────────────────────────────

    def update(
        self,
        blendshapes: Any,
        head_pose: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        """
        Call once per frame.
        blendshapes : MediaPipe blendshape list (or None if no face)
        head_pose   : (yaw_deg, pitch_deg, roll_deg, face_scale) or None
        """
        if head_pose is not None:
            self._has_pose = True
        vec = _extract(blendshapes, head_pose)
        self._buf.append(vec)
        self._compute()

    def render(self, canvas: np.ndarray) -> np.ndarray:
        """Draw onto a BGR canvas (any size ≥ 400×300). Returns canvas."""
        for s in self.STATES:
            self.smooth_scores[s] = (
                self.smooth_scores[s] * (1 - self.smooth_alpha)
                + self.target_scores[s] * self.smooth_alpha
            )
            self._history[s].append(self.smooth_scores[s])
        self._draw(canvas)
        return canvas

    # ── scoring engine ────────────────────────────────────────────────────────

    def _compute(self) -> None:
        valid = [v for v in self._buf if v is not None]
        n = len(valid)

        if n < self.min_frames:
            for s in self.STATES:
                self.target_scores[s] = 0.0
            self.dominant_state = "warming up…"
            self.dominant_conf  = 0.0
            return

        arr = np.stack(valid, axis=0)   # (n, n_features)
        st  = _Stats(arr)

        conf  = self._score_confusion(st, n)
        frus  = self._score_frustration(st, n)
        bore  = self._score_boredom(st, n)
        conc  = self._score_concentration(st, n)

        # ── Cross-suppression ──────────────────────────────────────────────
        # Frustration and boredom are physiologically incompatible at high levels
        frus = _clip01(frus - 0.35 * bore)
        bore = _clip01(bore - 0.45 * frus)
        # Confusion suppresses concentration
        conc = _clip01(conc - 0.40 * conf)
        # Frustration suppresses concentration
        conc = _clip01(conc - 0.30 * frus)
        # Boredom suppresses confusion slightly
        conf = _clip01(conf - 0.15 * bore)

        self.target_scores["Confusion"]     = conf
        self.target_scores["Frustration"]   = frus
        self.target_scores["Boredom"]       = bore
        self.target_scores["Concentration"] = conc

        best = max(self.STATES, key=lambda s: self.target_scores[s])
        self.dominant_state = best
        self.dominant_conf  = self.target_scores[best]

        # Cache active signal lists for HUD
        THRESH = 0.30
        self._active_conf_signals = [
            k for k, v in self.sub_conf.items() if v >= THRESH]
        self._active_frus_signals = [
            k for k, v in self.sub_frus.items() if v >= THRESH]

    # ── CONFUSION ─────────────────────────────────────────────────────────────

    def _score_confusion(self, st: _Stats, n: int) -> float:
        """
        7 sub-signals.  Weights sum to 1.0.
        Distinguishing confusion from frustration:
          confusion = inner-brow oblique raise + moderate furrow + tilt + lean
          frustration = HARD flat furrow + lip/nose tension (no tilt, no lean)
        """

        # [1] Inner brow raise (AU1 — oblique pull upward and inward)
        #     Threshold at 0.12 (subtle); score saturates at 0.40
        ib_raise = _clip01(st.m("browInnerUp") / 0.22)

        # [2] Moderate brow furrow (AU4) — present but NOT as hard as frustration
        #     Use a soft window: score peaks at 0.15, drops off above 0.28
        raw_furrow = st.avg_m("browDownLeft", "browDownRight")
        # Tent function: rises to 0.15, flat 0.15–0.22, falls after 0.28
        brow_mod = _clip01(raw_furrow / 0.15) * _clip01(1.0 - (raw_furrow - 0.22) / 0.12)
        brow_mod = _clip01(brow_mod)

        # [3] Brow + squint combo — the "screwed-up" confusion face
        #     Both browInnerUp AND eyeSquint must be simultaneously active
        squint   = st.avg_m("eyeSquintLeft", "eyeSquintRight")
        combo    = _clip01(ib_raise * 0.6 + squint * 0.4) * _clip01(ib_raise / 0.15)

        # [4] Brow asymmetry — one side furrows more (unilateral confusion)
        #     Robust: use mean absolute deviation not just one-frame diff
        brow_l   = st.col("browDownLeft")
        brow_r   = st.col("browDownRight")
        asym_per_frame = np.abs(brow_l - brow_r)
        brow_asym = _clip01(float(asym_per_frame.mean()) / 0.08)

        # [5] Head tilt — |roll| sustained above 8°
        #     Normalised roll stored as roll/45 → threshold = 8/45 ≈ 0.178
        roll_norm = np.abs(st.col("roll"))
        tilt_score = _clip01(float(roll_norm.mean()) / (8.0 / 45.0))
        if not self._has_pose:
            tilt_score = 0.0

        # [6] Lean-in — face_scale positive trend (face growing = getting closer)
        #     face_scale raw values ~0.30–0.55 depending on camera distance
        #     A trend of +0.001/frame over 90 frames = ~+0.09 total (noticeable)
        fscale_trend = st.t("face_scale")
        lean_in = _clip01(fscale_trend / 0.0015)   # saturates at fast lean
        if not self._has_pose:
            lean_in = 0.0

        # [7] Brow flicker — repeated micro-confusion pulses
        #     browInnerUp std high means it's going up and down repeatedly
        brow_flicker = _clip01(st.s("browInnerUp") / 0.08)

        subs = {
            "inner_brow":   (0.18, ib_raise),
            "brow_furrow":  (0.14, brow_mod),
            "squint_combo": (0.16, combo),
            "brow_asym":    (0.14, brow_asym),
            "head_tilt":    (0.16, tilt_score),
            "lean_in":      (0.14, lean_in),
            "brow_flicker": (0.08, brow_flicker),
        }
        self.sub_conf = {k: v for k, (_, v) in subs.items()}
        return _clip01(sum(w * v for _, (w, v) in subs.items()))

    # ── FRUSTRATION ───────────────────────────────────────────────────────────

    def _score_frustration(self, st: _Stats, n: int) -> float:
        """
        7 sub-signals.
        Key distinction from confusion: hard flat furrow, perioral tension,
        nasal wrinkle, and head restlessness — NOT tilt or lean-in.
        """

        # [1] Hard brow furrow — must be clearly above confusion range (>0.20)
        raw_furrow = st.avg_m("browDownLeft", "browDownRight")
        hard_furrow = _clip01((raw_furrow - 0.12) / 0.20)   # onset at 0.12

        # [2] Lip press (AU28) — lips compressed inward, strong frustration marker
        lip_press = _clip01(st.avg_m("mouthPressLeft", "mouthPressRight") / 0.15)

        # [3] Nose sneer (AU9) — nostril raise / upper-lip pull
        nose_sneer = _clip01(st.avg_m("noseSneerLeft", "noseSneerRight") / 0.10)

        # [4] Lip corner pull-down (AU15) — mouth frown
        lip_frown = _clip01(st.avg_m("mouthFrownLeft", "mouthFrownRight") / 0.14)

        # [5] Jaw tension — mouthClose high without jawOpen
        #     When jaw is clenched: mouthClose rises, jawOpen stays low
        jaw_close   = st.m("mouthClose")
        jaw_open_mu = st.m("jawOpen")
        jaw_tension = _clip01(jaw_close / 0.20) * _clip01(1.0 - jaw_open_mu / 0.15)

        # [6] Head agitation — high positional variance in yaw + pitch
        #     Restless frustration: head shifts back and forth
        yaw_var   = st.s("yaw")
        pitch_var = st.s("pitch")
        head_agit = _clip01((yaw_var + pitch_var) / (0.08 + 0.06))
        if not self._has_pose:
            head_agit = 0.0

        # [7] Escalation — frustration proxy (hard_furrow + lip_press) trending up
        #     over the whole window: things are getting worse
        frus_proxy = (
            st.col("browDownLeft") * 0.3 +
            st.col("browDownRight") * 0.3 +
            st.col("mouthPressLeft") * 0.2 +
            st.col("mouthPressRight") * 0.2
        )
        escalation = _clip01(_lintrend(frus_proxy) / 0.002)

        subs = {
            "hard_furrow":  (0.22, hard_furrow),
            "lip_press":    (0.20, lip_press),
            "nose_sneer":   (0.15, nose_sneer),
            "lip_frown":    (0.13, lip_frown),
            "jaw_tension":  (0.12, jaw_tension),
            "head_agit":    (0.12, head_agit),
            "escalation":   (0.06, escalation),
        }
        self.sub_frus = {k: v for k, (_, v) in subs.items()}
        return _clip01(sum(w * v for _, (w, v) in subs.items()))

    # ── BOREDOM ───────────────────────────────────────────────────────────────

    # ── BOREDOM ───────────────────────────────────────────────────────────────

    def _score_boredom(self, st: _Stats, n: int) -> float:
        """
        7 sub-signals for Boredom based on affective computing heuristics:
        Includes eyes looking down, ptosis (heavy eyelids), slack face/jaw,
        physical distancing, and head drop.
        """

        # [1] Eyes looking DOWN — gaze cast below the screen
        eyes_down = _clip01(st.avg_m("eyeLookDownLeft", "eyeLookDownRight") / 0.40)

        # [2] Heavy Eyelids (Ptosis) — partial closure, slowed blinks
        # eyeBlink normally near 0.0. If hovering around 0.15-0.40, lids are drooping.
        # Score decreases if it goes too high (which means eyes are fully shut/sleeping)
        blinks = st.avg_m("eyeBlinkLeft", "eyeBlinkRight")
        heavy_eyelids = _clip01(blinks / 0.15) * _clip01(1.0 - (blinks - 0.30) / 0.40)

        # [3] Face slack — overall AU activity (excluding blinks/gaze) is very low
        expression_keys = [k for k in _BS_KEYS if not k.startswith("eyeLook") and "Blink" not in k]
        expr_var = float(np.mean([st.col(k).std() for k in expression_keys]))
        expr_mu  = float(np.mean([st.m(k) for k in expression_keys]))
        face_slack = _clip01(1.0 - expr_var / 0.055) * _clip01(1.0 - expr_mu / 0.08)

        # [4] Mouth slack — jaw loosely hanging (not clenched, not yawning)
        mouth_close_mu = st.m("mouthClose")
        jaw_open_mu    = st.m("jawOpen")
        lip_press_mu   = st.avg_m("mouthPressLeft", "mouthPressRight")
        mouth_slack = (
            _clip01(1.0 - mouth_close_mu / 0.30) * _clip01(1.0 - jaw_open_mu    / 0.20) * _clip01(1.0 - lip_press_mu   / 0.10)
        )

        # [5] Far from screen — face_scale is small (leaning back in chair)
        if self._has_pose:
            fscale_mu = st.m("face_scale")
            far_from_screen = _clip01((0.38 - fscale_mu) / 0.18)
        else:
            far_from_screen = 0.0

        # [6] Head drop (Chin Drop) — pitch is significantly positive (tilted down)
        # Normalised pitch: 1.0 = 30 degrees. Score rises as head drops below 10 degrees.
        if self._has_pose:
            pitch_mu = st.m("pitch")
            head_drop = _clip01(pitch_mu / 0.40) 
        else:
            head_drop = 0.0

        # [7] Stillness (Lack of micro-nods) — highly rigid/slumped posture
        if self._has_pose:
            head_still = _clip01(1.0 - (st.s("yaw") + st.s("pitch")) / 0.07)
        else:
            head_still = 0.0

        # Weights sum to 1.0
        subs = {
            "eyes_down":      (0.20, eyes_down),
            "heavy_eyelids":  (0.15, heavy_eyelids),
            "face_slack":     (0.15, face_slack),
            "mouth_slack":    (0.15, mouth_slack),
            "far_screen":     (0.15, far_from_screen),
            "head_drop":      (0.10, head_drop),
            "head_still":     (0.10, head_still),
        }
        return _clip01(sum(w * v for w, v in subs.values()))
    # ── CONCENTRATION ─────────────────────────────────────────────────────────

    def _score_concentration(self, st: _Stats, n: int) -> float:
        # Mild furrow: slight browDown, not full distress
        raw_furrow   = st.avg_m("browDownLeft", "browDownRight")
        mild_furrow  = _clip01(raw_furrow / 0.12) * _clip01(1.0 - raw_furrow / 0.22)
        # Mouth still: low variance → not talking or grimacing
        mouth_still  = _clip01(1.0 - st.avg_s("mouthSmileLeft", "mouthFrownLeft") / 0.05)
        # Eyes open: low blink rate
        eyes_open    = _clip01(1.0 - st.avg_m("eyeBlinkLeft", "eyeBlinkRight") / 0.15)
        # Head stable: low pose variance → not fidgeting
        if self._has_pose:
            head_stable = _clip01(1.0 - (st.s("yaw") + st.s("pitch")) / 0.10)
        else:
            head_stable = 0.5   # unknown, neutral assumption
        # Upright posture: |pitch| low and |roll| low
        if self._has_pose:
            upright = _clip01(1.0 - abs(st.m("pitch")) / 0.30) * \
                      _clip01(1.0 - abs(st.m("roll")) / 0.25)
        else:
            upright = 0.5

        subs = {
            "mild_furrow": (0.20, mild_furrow),
            "mouth_still": (0.20, mouth_still),
            "eyes_open":   (0.25, eyes_open),
            "head_stable": (0.20, head_stable),
            "upright":     (0.15, upright),
        }
        return _clip01(sum(w * v for w, v in subs.values()))

    # ── RENDERING ─────────────────────────────────────────────────────────────

    def _draw(self, img: np.ndarray) -> None:
        H, W = img.shape[:2]
        img[:] = (16, 16, 20)

        # ── header ────────────────────────────────────────────────────────
        self._t(img, "EPISTEMIC  STATE", (14, 28), 0.62, (170, 170, 170), 1)

        valid_n = sum(1 for v in self._buf if v is not None)
        pct = min(valid_n / max(self.min_frames, 1), 1.0)
        bw  = int((W - 28) * pct)
        cv2.rectangle(img, (14, 35), (14 + bw, 39), (55, 90, 55), -1)
        cv2.rectangle(img, (14, 35), (W - 14, 39), (45, 45, 45), 1)
        secs = valid_n / self.fps
        self._t(img, f"{secs:.0f}s / {self.window_frames // self.fps}s",
                (W - 68, 28), 0.38, (70, 70, 70))
        if self._has_pose:
            self._t(img, "● POSE", (W - 68, 46), 0.36, (60, 120, 60))

        # ── dominant badge ────────────────────────────────────────────────
        dom_c = self._COLORS.get(self.dominant_state, (140, 140, 140))
        if self.dominant_conf > 0.12:
            bx, by = 14, 48
            cv2.rectangle(img, (bx, by), (bx + 230, by + 30), (26, 26, 30), -1)
            cv2.rectangle(img, (bx, by), (bx + 230, by + 30), dom_c, 1)
            self._t(img, self.dominant_state.upper(), (bx + 7, by + 20),
                    0.60, dom_c, 2)
            self._t(img, f"{self.dominant_conf:.0%}",
                    (bx + 168, by + 20), 0.52, (175, 175, 175))

        # ── state bars ────────────────────────────────────────────────────
        BX    = 136
        BW    = W - BX - 14
        ROW   = 70
        SY    = 88

        for i, state in enumerate(self.STATES):
            y     = SY + i * ROW
            score = self.smooth_scores[state]
            col   = self._COLORS[state]
            bl    = int(BW * score)

            self._t(img, state,           (10, y + 15), 0.52, (195, 195, 195), 1)
            self._t(img, self._hint(state),(10, y + 29), 0.34, (75, 75, 75))

            cv2.rectangle(img, (BX, y + 2), (BX + BW, y + 22), (32, 32, 38), -1)
            if bl > 0:
                cv2.rectangle(img, (BX, y + 2), (BX + bl, y + 22), col, -1)
                if score > 0.55:
                    glow = tuple(min(255, int(c * 1.35)) for c in col)
                    cv2.rectangle(img, (BX, y + 2), (BX + bl, y + 12), glow, -1)

            lbl_x = min(BX + bl + 5, W - 44)
            self._t(img, f"{score:.2f}", (lbl_x, y + 17), 0.44, col)

            hist = list(self._history[state])
            if len(hist) > 4:
                self._spark(img, hist, BX, y + 26, BW, 24, col)

        # ── sub-signal panels (Confusion + Frustration) ───────────────────
        panel_y = SY + 4 * ROW + 6
        panel_h = H - panel_y - 8
        mid     = W // 2

        self._sub_panel(img, "CONFUSION SIGNALS", self.sub_conf,
                        2, panel_y, mid - 4, panel_h,
                        self._COLORS["Confusion"])
        self._sub_panel(img, "FRUSTRATION SIGNALS", self.sub_frus,
                        mid + 2, panel_y, W - mid - 4, panel_h,
                        self._COLORS["Frustration"])

    def _sub_panel(self, img, title, subs, x, y, w, h, col):
        if not subs or h < 20:
            return
        cv2.rectangle(img, (x, y), (x + w, y + h), (24, 24, 28), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 1)
        self._t(img, title, (x + 4, y + 11), 0.32, col, 1)
        n   = len(subs)
        row = max(1, (h - 16) // n)
        for j, (name, val) in enumerate(subs.items()):
            ry  = y + 16 + j * row
            bw  = int((w - 8) * val)
            dim = tuple(max(0, int(c * 0.45)) for c in col)
            cv2.rectangle(img, (x + 4, ry), (x + w - 4, ry + row - 2), (28, 28, 32), -1)
            if bw > 0:
                cv2.rectangle(img, (x + 4, ry), (x + 4 + bw, ry + row - 2), dim, -1)
            active = val >= 0.30
            tc = col if active else (70, 70, 70)
            label = f"{'●' if active else '○'} {name}"
            self._t(img, label, (x + 6, ry + row - 4), 0.30, tc)

    @staticmethod
    def _hint(state: str) -> str:
        return {
            "Concentration": "mild furrow · still · eyes open",
            "Confusion":     "inner brow · tilt · lean-in",
            "Boredom":       "eyes down · heavy lids · slack jaw · head drop", # UPDATED
            "Frustration":   "hard furrow · lip press · agitation",
        }.get(state, "")

    @staticmethod
    def _t(img, text, pos, scale=0.50, color=(200, 200, 200), thickness=1):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _spark(img, values, x0, y0, w, h, color):
        v = np.array(values, dtype=np.float32)
        lo, hi = v.min(), v.max()
        if hi - lo < 0.01:
            return
        norm = (v - lo) / (hi - lo)
        n = len(norm)
        pts = [(x0 + int(i / (n - 1) * w), y0 + h - int(norm[i] * h))
               for i in range(n)]
        for j in range(len(pts) - 1):
            alpha = 0.35 + 0.65 * (j / n)
            c = tuple(int(ch * alpha) for ch in color)
            cv2.line(img, pts[j], pts[j + 1], c, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    class BS:
        def __init__(self, n, s): self.category_name = n; self.score = s

    def confusion_bs():
        return [BS("browInnerUp", 0.38), BS("browDownLeft", 0.16),
                BS("browDownRight", 0.08), BS("eyeSquintLeft", 0.28),
                BS("eyeSquintRight", 0.25)]

    def frustration_bs():
        return [BS("browDownLeft", 0.45), BS("browDownRight", 0.43),
                BS("mouthPressLeft", 0.32), BS("mouthPressRight", 0.30),
                BS("noseSneerLeft", 0.20), BS("mouthFrownLeft", 0.22),
                BS("mouthClose", 0.25)]

    for label, bs_fn, pose in [
        ("CONFUSION    (+ tilt + lean)",
         confusion_bs, (2.0, -3.0, 14.0, 0.38)),   # roll=14°, leaning in
        ("FRUSTRATION  (+ agitation)",
         frustration_bs, (0.0, 0.0, 1.0, 0.32)),    # minimal tilt
    ]:
        t = LocalEpistemicTracker(window_frames=60, min_frames=10)
        import numpy as np
        rng = np.random.default_rng(0)
        for f in range(60):
            # Simulate some head agitation for frustration
            p = (pose[0] + rng.uniform(-2, 2),
                 pose[1] + rng.uniform(-1, 1),
                 pose[2] + rng.uniform(-1, 1),
                 pose[3] + rng.uniform(-0.002, 0.002) * f)
            t.update(bs_fn(), head_pose=p)
        t.render(np.zeros((540, 640, 3), dtype=np.uint8))
        print(f"\n{label}")
        print(f"  dominant : {t.dominant_state}  {t.dominant_conf:.0%}")
        print(f"  scores   : { {k: f'{v:.2f}' for k,v in t.smooth_scores.items()} }")
        print(f"  conf subs: { {k: f'{v:.2f}' for k,v in t.sub_conf.items()} }")
        print(f"  frus subs: { {k: f'{v:.2f}' for k,v in t.sub_frus.items()} }")