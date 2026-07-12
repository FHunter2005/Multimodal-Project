from collections import deque
from typing import Optional, Tuple, Any
import cv2
import numpy as np


def _lintrend(arr: np.ndarray) -> float:
    n = len(arr)
    if n < 4:
        return 0.0
    x = np.arange(n, dtype=np.float32) - (n - 1) / 2.0
    denom = float((x * x).sum())
    if denom < 1e-9:
        return 0.0
    return float((x * (arr - arr.mean())).sum() / denom)


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
    "noseSneerLeft",   "noseSneerRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookUpLeft",   "eyeLookUpRight",
]

_POSE_KEYS = ["yaw", "pitch", "roll", "face_scale"]
_ALL_KEYS  = _BS_KEYS + _POSE_KEYS
_KEY_IDX   = {k: i for i, k in enumerate(_ALL_KEYS)}


def _extract(blendshapes, head_pose: Optional[Tuple]) -> Optional[np.ndarray]:
    if not blendshapes:
        return None
    bs = {b.category_name: float(b.score) for b in blendshapes}
    vec = np.array([bs.get(k, 0.0) for k in _BS_KEYS], dtype=np.float32)

    if head_pose is not None:
        yaw, pitch, roll, fscale = head_pose
        pose_vec = np.array([
            float(yaw)    / 45.0,   
            float(pitch)  / 30.0,
            float(roll)   / 45.0,
            float(fscale),          
        ], dtype=np.float32)
    else:
        pose_vec = np.zeros(len(_POSE_KEYS), dtype=np.float32)

    return np.concatenate([vec, pose_vec])


def _i(key: str) -> int:
    return _KEY_IDX[key]


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


class _Stats:
    def __init__(self, arr: np.ndarray):
        self.arr   = arr                          
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

class LocalEpistemicTracker:
    STATES = ["concentration", "confusion", "frustration"]

    _COLORS = {
        "concentration": (200, 175,  50),
        "confusion":     ( 50, 170, 255),
        "frustration":   ( 40,  50, 215),
    }

    def __init__(
        self,
        window_frames: int   = 90,
        smooth_alpha:  float = 0.03,
        min_frames:    int   = 20,
        fps:           int   = 30,
        escalation_t_sec: float = 3.0,     
        escalation_rate:  float = 0.15,    
        escalation_thresh: float = 0.35
    ):
        self.window_frames = window_frames
        self.smooth_alpha  = smooth_alpha
        self.min_frames    = min_frames
        self.fps           = fps
        self.escalation_t_sec = escalation_t_sec
        self.escalation_rate = escalation_rate
        self.escalation_thresh = escalation_thresh
        self.active_frames = {s: 0 for s in self.STATES}
        self._buf: deque = deque(maxlen=window_frames)   
        self._has_pose   = False                         

        self.target_scores = {s: 0.0 for s in self.STATES}
        self.smooth_scores = {s: 0.0 for s in self.STATES}
        self.current_state = {s: 0.0 for s in self.STATES}

        self.sub_conf = {}   
        self.sub_frus = {}

        _hl = 150
        self._history = {s: deque([0.0] * _hl, maxlen=_hl) for s in self.STATES}

        self.dominant_state = "—"
        self.dominant_conf  = 0.0

        self._active_conf_signals: list = []
        self._active_frus_signals: list = []

    def update(self, blendshapes: Any, head_pose: Optional[Tuple[float, float, float, float]] = None) -> None:
        if head_pose is not None:
            self._has_pose = True
        vec = _extract(blendshapes, head_pose)
        self._buf.append(vec)
        self._compute()

    def render(self, canvas: np.ndarray) -> np.ndarray:
        for s in self.STATES:
            self.smooth_scores[s] = (
                self.smooth_scores[s] * (1 - self.smooth_alpha)
                + self.target_scores[s] * self.smooth_alpha
            )
            self._history[s].append(self.smooth_scores[s])
        self._draw(canvas)
        return canvas

    def _compute(self) -> None:
        valid = [v for v in self._buf if v is not None]
        n = len(valid)

        if n < self.min_frames:
            for s in self.STATES:
                self.target_scores[s] = 0.0
            self.dominant_state = "warming up…"
            self.dominant_conf  = 0.0
            return

        arr = np.stack(valid, axis=0)
        st  = _Stats(arr)

        conf  = self._score_confusion(st, n)
        frus  = self._score_frustration(st, n)
        conc  = self._score_concentration(st, n)

        conc = _clip01(conc - 0.40 * conf)
        conc = _clip01(conc - 0.50 * frus)
        conf = _clip01(conf - 0.20 * frus) 
        base_scores = {
            "confusion": conf,
            "frustration": frus,
            "concentration": conc
        }

        escalating_states = ["confusion", "frustration"]

        for s in self.STATES:
            if base_scores[s] >= self.escalation_thresh:
                self.active_frames[s] += 1
            else:
                self.active_frames[s] = max(0, self.active_frames[s] - 2)

            if s in escalating_states:
                # Cap accumulated "active" time: without this, sustained
                # confusion/frustration escalates forever and, since decay
                # is only -2/frame, can take minutes to unwind even long
                # after the expression that triggered it is gone.
                max_active_frames = (self.escalation_t_sec + 6.0) * self.fps
                self.active_frames[s] = min(self.active_frames[s], max_active_frames)

                frames_past_t = self.active_frames[s] - (self.escalation_t_sec * self.fps)
                if frames_past_t > 0:
                    bonus = (frames_past_t / self.fps) * self.escalation_rate
                    base_scores[s] = _clip01(base_scores[s] + bonus)

        self.target_scores["confusion"]     = base_scores["confusion"]
        self.target_scores["frustration"]   = base_scores["frustration"]
        self.target_scores["concentration"] = base_scores["concentration"]

        self.current_state = dict(self.target_scores)

        best = max(self.STATES, key=lambda s: self.target_scores[s])
        self.dominant_state = best
        self.dominant_conf  = self.target_scores[best]

        THRESH = 0.30
        self._active_conf_signals = [k for k, v in self.sub_conf.items() if v >= THRESH]
        self._active_frus_signals = [k for k, v in self.sub_frus.items() if v >= THRESH]

    def _score_confusion(self, st: _Stats, n: int) -> float:
        """
        FACS MAPPING (AU1, AU2, AU4):
        Tracking the "Quizzical" look where brows go UP (raise) and DOWN (furrow) 
        simultaneously or asymmetrically.
        """
        # [1] AU1 + AU2: Inner and Outer brow raise
        brow_raise = _clip01(st.avg_m("browInnerUp", "browOuterUpLeft", "browOuterUpRight") / 0.25)

        # [2] AU4: Brow furrow (Brows going down). Moderate activation.
        brow_down = st.avg_m("browDownLeft", "browDownRight")
        brow_furrow = _clip01(brow_down / 0.20)

        # [3] The Quizzical Combo: Brows going UP and DOWN simultaneously.
        # Multiplies inner raise by overall furrow. High score only if BOTH are active.
        quizzical = _clip01((st.m("browInnerUp") * brow_down) * 15.0) 

        # [4] Brow Asymmetry: One brow up, one brow down.
        asym = _clip01(float(np.abs(st.col("browDownLeft") - st.col("browDownRight")).mean()) / 0.10)

        # [5] Posture: Head Tilt (Roll) & Lean In
        tilt = _clip01(float(np.abs(st.col("roll")).mean()) / (8.0 / 45.0)) if self._has_pose else 0.0
        lean = _clip01(st.t("face_scale") / 0.0015) if self._has_pose else 0.0

        subs = {
            "brow_raise": (0.20, brow_raise),
            "brow_down":  (0.10, brow_furrow),
            "quizzical":  (0.25, quizzical),
            "asymmetry":  (0.20, asym),
            "head_tilt":  (0.15, tilt),
            "lean_in":    (0.10, lean),
        }
        self.sub_conf = {k: v for k, (_, v) in subs.items()}
        return _clip01(sum(w * v for _, (w, v) in subs.items()))

    def _score_frustration(self, st: _Stats, n: int) -> float:
        """
        FACS MAPPING (AU4, AU7, AU24):
        Hard flat furrow, perioral tension (lips), eye squint, and head agitation.
        """
        # [1] AU4: Hard Brow Furrow (Stronger baseline required than confusion)
        brow_down = st.avg_m("browDownLeft", "browDownRight")
        hard_furrow = _clip01((brow_down - 0.10) / 0.25) 

        # [2] AU24: Lip press — lips compressed inward, strong frustration/anger marker
        lip_press = _clip01(st.avg_m("mouthPressLeft", "mouthPressRight") / 0.18)

        # [3] AU7: Lid Tightener / Squint — common in aggressive/frustrated focus
        squint = _clip01(st.avg_m("eyeSquintLeft", "eyeSquintRight") / 0.25)

        # [4] AU9: Nose sneer — disgust/frustration wrinkle
        sneer = _clip01(st.avg_m("noseSneerLeft", "noseSneerRight") / 0.15)

        # [5] Jaw tension — mouthClose high without jawOpen
        jaw_tension = _clip01(st.m("mouthClose") / 0.20) * _clip01(1.0 - st.m("jawOpen") / 0.15)

        # [6] Head agitation — high positional variance in yaw + pitch
        head_agit = _clip01((st.s("yaw") + st.s("pitch")) / 0.12) if self._has_pose else 0.0

        subs = {
            "hard_furrow": (0.25, hard_furrow),
            "lip_press":   (0.20, lip_press),
            "eye_squint":  (0.15, squint),
            "nose_sneer":  (0.15, sneer),
            "jaw_tension": (0.10, jaw_tension),
            "head_agit":   (0.15, head_agit),
        }
        self.sub_frus = {k: v for k, (_, v) in subs.items()}
        return _clip01(sum(w * v for _, (w, v) in subs.items()))



    def _score_concentration(self, st: _Stats, n: int) -> float:
        raw_furrow   = st.avg_m("browDownLeft", "browDownRight")
        mild_furrow  = _clip01(raw_furrow / 0.08) * _clip01(1.0 - (raw_furrow - 0.16) / 0.15)
        
        mouth_still  = _clip01(1.0 - st.avg_s("mouthSmileLeft", "mouthFrownLeft") / 0.05)
        eyes_open    = _clip01(1.0 - st.avg_m("eyeBlinkLeft", "eyeBlinkRight") / 0.15)
        
        if self._has_pose:
            head_stable = _clip01(1.0 - (st.s("yaw") + st.s("pitch")) / 0.10)
            upright = _clip01(1.0 - abs(st.m("pitch")) / 0.30) * \
                      _clip01(1.0 - abs(st.m("roll")) / 0.25)
        else:
            head_stable = 0.5
            upright = 0.5

        subs = {
            "mild_furrow": (0.45, mild_furrow), 
            "mouth_still": (0.10, mouth_still), 
            "eyes_open":   (0.15, eyes_open),   
            "head_stable": (0.15, head_stable), 
            "upright":     (0.15, upright),     
        }
        return _clip01(sum(w * v for w, v in subs.values()))



    def _draw(self, img: np.ndarray) -> None:
        H, W = img.shape[:2]
        img[:] = (16, 16, 20)

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

        dom_c = self._COLORS.get(self.dominant_state, (140, 140, 140))
        if self.dominant_conf > 0.12:
            bx, by = 14, 48
            cv2.rectangle(img, (bx, by), (bx + 230, by + 30), (26, 26, 30), -1)
            cv2.rectangle(img, (bx, by), (bx + 230, by + 30), dom_c, 1)
            self._t(img, self.dominant_state.upper(), (bx + 7, by + 20),
                    0.60, dom_c, 2)
            self._t(img, f"{self.dominant_conf:.0%}",
                    (bx + 168, by + 20), 0.52, (175, 175, 175))

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

        panel_y = SY + 3 * ROW + 6
        panel_h = H - panel_y - 8
        mid     = W // 2

        self._sub_panel(img, "CONFUSION SIGNALS", self.sub_conf,
                        2, panel_y, mid - 4, panel_h,
                        self._COLORS["confusion"])
        self._sub_panel(img, "FRUSTRATION SIGNALS", self.sub_frus,
                        mid + 2, panel_y, W - mid - 4, panel_h,
                        self._COLORS["frustration"])

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
            "concentration": "mild furrow · still · eyes open",
            "confusion":     "quizzical brows · asymmetry · tilt",
            "frustration":   "hard furrow · lip press · agitation",
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