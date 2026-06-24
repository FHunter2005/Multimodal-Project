"""
PDF Gaze Reader — ResNet eye-crop edition
==========================================
Eye tracking : RetinaFace (face + eye landmarks) +
               frozen ResNet50 on EACH EYE SEPARATELY (2×2048-D) +
               PCA + Ridge calibration
PDF reading  : PyMuPDF, paragraph detection, stuck detector, summary dialog

Install
-------
  pip install torch torchvision opencv-python pymupdf scikit-learn anthropic
  pip install git+https://github.com/Ahmednull/L2CS-Net.git   # provides RetinaFace

Why eye crops beat face crops
------------------------------
  Full face: 2048-D ResNet features encode hair, background, skin tone —
             mostly irrelevant to gaze direction.
  Eye crops: each 112×112 crop is centred on the iris. ResNet features
             now capture iris position + eyelid shape, which are the actual
             signals used for gaze estimation. Calibration maps this richer
             4096-D space (left eye ∥ right eye) to screen coordinates.

Controls
--------
  SPACE   confirm calibration dot
  n/p     next/prev page
  d       drift correction
  r       recalibrate
  q       quit
"""

import cv2, time, numpy as np, threading, os, sys, argparse
from collections import deque
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

try:
    import fitz
except ImportError:
    print("[ERROR] pip install pymupdf"); sys.exit(1)

try:
    from face_detection import RetinaFace
except ImportError:
    print("[ERROR] pip install git+https://github.com/Ahmednull/L2CS-Net.git"); sys.exit(1)

SCREEN_W = 1920
SCREEN_H  = 1080

# ── Kalman ────────────────────────────────────────────────────
class KalmanGaze:
    def __init__(self, pn=1e-3, mn=3e-2, dt=1/30):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix    = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        self.kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * pn
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * mn
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self._ok = False
    def update(self, x, y):
        if not self._ok:
            self.kf.statePre = self.kf.statePost = np.array([[x],[y],[0],[0]], dtype=np.float32)
            self._ok = True; return x, y
        self.kf.predict()
        c = self.kf.correct(np.array([[x],[y]], dtype=np.float32))
        return float(c[0,0]), float(c[1,0])
    def reset(self): self._ok = False

# ── Async gaze estimator (eye-crop edition) ───────────────────
class GazeEstimator:
    """
    RetinaFace gives face box + 5 facial landmarks (including both eye centres).
    We crop a square around each eye, extract 2048-D ResNet50 features from each,
    and concatenate → 4096-D feature vector used for calibration.
    Runs in a background thread; UI never blocks on CNN inference.
    """
    # Eye crops are small — resize directly to 112×112 (no centre-crop needed)
    _transform = T.Compose([
        T.ToPILImage(),
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, device):
        gpu_id = -1 if device.type == 'cpu' else (device.index or 0)
        self._detector = RetinaFace(gpu_id=gpu_id)

        # Frozen ResNet50 — global avg pool collapses any spatial size → 2048-D
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self._extractor = nn.Sequential(*list(resnet.children())[:-1])
        self._extractor.eval()
        for p in self._extractor.parameters():
            p.requires_grad_(False)
        self._extractor.to(device)
        self._device = device

        self._lock    = threading.Lock()
        self._pending = None
        self.features = None      # shape (4096,): left-eye ∥ right-eye features
        self.eye_pts  = None      # ((ex0,ey0), (ex1,ey1)) in raw-frame coords
        self.detected = False
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, frame):
        with self._lock:
            self._pending = frame

    def _crop_eye(self, frame, cx, cy, size):
        """Square crop of `size` pixels centred on (cx, cy)."""
        half = size // 2
        h, w = frame.shape[:2]
        x1 = max(0, int(cx - half)); y1 = max(0, int(cy - half))
        x2 = min(w, int(cx + half)); y2 = min(h, int(cy + half))
        crop = frame[y1:y2, x1:x2]
        return crop if (crop.shape[0] >= 8 and crop.shape[1] >= 8) else None

    def _extract(self, crop_bgr):
        """BGR crop → 2048-D ResNet50 feature vector."""
        rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feat = self._extractor(tensor)
        return feat.squeeze().cpu().numpy()   # (2048,)

    def _run(self):
        while self._running:
            frame = None
            with self._lock:
                if self._pending is not None:
                    frame, self._pending = self._pending, None
            if frame is None:
                time.sleep(0.005); continue
            try:
                faces = self._detector(frame)
                if faces is not None and len(faces) > 0:
                    box, landmark, score = faces[0]
                    if score >= 0.5:
                        # RetinaFace landmarks: [eye_a, eye_b, nose, mouth_a, mouth_b]
                        # (which is left/right depends on orientation — we use both)
                        e0, e1 = landmark[0], landmark[1]
                        face_w  = box[2] - box[0]
                        eye_sz  = max(int(face_w * 0.42), 40)

                        c0 = self._crop_eye(frame, e0[0], e0[1], eye_sz)
                        c1 = self._crop_eye(frame, e1[0], e1[1], eye_sz)

                        if c0 is not None and c1 is not None:
                            feat = np.concatenate([self._extract(c0),
                                                   self._extract(c1)])  # (4096,)
                            with self._lock:
                                self.features = feat
                                self.eye_pts  = ((int(e0[0]), int(e0[1])),
                                                 (int(e1[0]), int(e1[1])))
                                self.detected = True
                            continue
            except Exception:
                pass
            with self._lock:
                self.detected = False

    def read(self):
        with self._lock:
            return self.features, self.eye_pts, self.detected

    def stop(self):
        self._running = False

# ── Calibration: 4096-D eye features → normalised screen coords
# PCA first (can't have more components than calibration points − 1),
# then Ridge with moderate alpha to avoid shrinking back to centre.
_cal_x  = None
_cal_y  = None
_cal_ok = False

def fit_calibration(features, targets):
    global _cal_x, _cal_y, _cal_ok
    X = np.array(features); Y = np.array(targets)
    n_comp = min(len(X) - 1, 50)
    _cal_x = make_pipeline(PCA(n_components=n_comp), Ridge(alpha=10.0))
    _cal_y = make_pipeline(PCA(n_components=n_comp), Ridge(alpha=10.0))
    _cal_x.fit(X, Y[:, 0])
    _cal_y.fit(X, Y[:, 1])
    _cal_ok = True

def predict_gaze(features):
    if not _cal_ok or features is None: return 0.5, 0.5
    f = features.reshape(1, -1)
    return (float(np.clip(_cal_x.predict(f)[0], 0, 1)),
            float(np.clip(_cal_y.predict(f)[0], 0, 1)))

# ── Camera thread ─────────────────────────────────────────────
class WebcamStream:
    def __init__(self, src=0, w=1280, h=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        _, self.frame = self.cap.read(); self.stopped = False; self.fid = 0
    def start(self):
        threading.Thread(target=self._run, daemon=True).start(); return self
    def _run(self):
        while not self.stopped:
            ok, f = self.cap.read()
            if ok: self.frame = f; self.fid += 1
    def read(self): return self.frame
    def stop(self): self.stopped = True; self.cap.release()

# ── PDF ───────────────────────────────────────────────────────
class PDFDocument:
    def __init__(self, path, zoom=None):
        self.doc = fitz.open(path)
        self.n = len(self.doc)
        # Auto-fit: scale so page width fills ~95 % of screen width
        if zoom is None:
            pw = self.doc[0].rect.width
            self.zoom = (SCREEN_W * 0.95) / pw
        else:
            self.zoom = zoom
        self._cache = {}
    def get_page(self, num):
        if num in self._cache: return self._cache[num]
        page = self.doc[num]
        pix  = page.get_pixmap(matrix=fitz.Matrix(self.zoom, self.zoom), alpha=False)
        img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        img  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ph   = page.rect.height; paras = []
        for b in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
            if b["type"] != 0: continue
            x0,y0,x1,y1 = [c*self.zoom for c in b["bbox"]]
            text = " ".join(s["text"] for l in b.get("lines",[])
                            for s in l.get("spans",[])).strip()
            if len(text) < 80 or b["bbox"][1] < ph*0.12: continue
            lines = b.get("lines", [])
            if len(lines) <= 2:
                avg = np.mean([s.get("size",10) for l in lines
                               for s in l.get("spans",[])] or [10])
                if avg > 13: continue
            paras.append({"bbox": (int(x0),int(y0),int(x1),int(y1)), "text": text})
        self._cache[num] = (img, paras); return img, paras
    def close(self): self.doc.close()

# ── Stuck detector ────────────────────────────────────────────
class StuckDetector:
    def __init__(self, dwell=5., cooldown=20.):
        self.dwell = dwell; self.cooldown = cooldown
        self._cur = None; self._t0 = None; self._fired = {}; self._fv = False
    def update(self, idx):
        now = time.time()
        if idx != self._cur:
            self._cur = idx; self._t0 = now if idx is not None else None
            self._fv = False; return None
        if idx is None or self._fv: return None
        if now-self._t0 >= self.dwell and now-self._fired.get(idx,0) >= self.cooldown:
            self._fired[idx] = now; self._fv = True; return idx
        return None
    def reset(self): self._cur=None; self._t0=None; self._fv=False; self._fired.clear()

# ── Summarizer ────────────────────────────────────────────────
def summarize(text):
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        try:
            from google import genai
            client = genai.Client(api_key=key)
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Summarize in 2-3 simple sentences for a student struggling with this:\n\n" + text)
            return resp.text.strip()
        except Exception as e: return f"[API error: {e}]"
    return "[Set GEMINI_API_KEY env var for AI summary]\n\n" + text[:400]

# ── UI helpers ────────────────────────────────────────────────
def draw_dot(canvas, tx, ty, prog, label):
    cv2.ellipse(canvas, (tx,ty), (22,22), -90, 0, int(360*prog), (0,255,100), 3)
    cv2.circle(canvas, (tx,ty), 12, (0,200,80), -1)
    tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
    cv2.putText(canvas, label, ((SCREEN_W-tw)//2, SCREEN_H//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)

def wrap_text(text, n=65):
    words = text.split(); lines = []; line = ""
    for w in words:
        if len(line)+len(w)+1 > n:
            if line: lines.append(line)
            line = w
        else: line = (line+" "+w).strip()
    if line: lines.append(line)
    return lines

AA = cv2.LINE_AA
F  = cv2.FONT_HERSHEY_DUPLEX   # smoother than SIMPLEX

def draw_dialog(canvas, dlg_summary):
    dim = canvas.copy(); canvas[:] = canvas // 3
    cv2.addWeighted(dim, 0.15, canvas, 0.85, 0, canvas)
    dw, dh = 900, 500; dx0=(SCREEN_W-dw)//2; dy0=(SCREEN_H-dh)//2
    cv2.rectangle(canvas, (dx0,dy0),    (dx0+dw,dy0+dh), (25,25,35),    -1)
    cv2.rectangle(canvas, (dx0,dy0),    (dx0+dw,dy0+dh), (100,180,255),  2)
    cv2.rectangle(canvas, (dx0,dy0),    (dx0+dw,dy0+52), (40,40,60),    -1)
    cv2.putText(canvas, "Looks like you might be stuck on this paragraph.",
                (dx0+20, dy0+36), F, 0.85, (100,210,255), 1, AA)
    cv2.line(canvas, (dx0, dy0+52), (dx0+dw, dy0+52), (100,180,255), 1)
    sumtext = dlg_summary['text']
    if sumtext is None:
        cv2.putText(canvas, "Generating summary...", (dx0+24, dy0+110),
                    F, 0.75, (160,160,160), 1, AA)
    else:
        for i, line in enumerate(wrap_text(sumtext)[:10]):
            cv2.putText(canvas, line, (dx0+24, dy0+96+i*36),
                        F, 0.72, (230,230,230), 1, AA)
    cv2.rectangle(canvas, (dx0+24,   dy0+dh-64), (dx0+220,  dy0+dh-18), (0,160,70),   -1)
    cv2.putText(canvas, "Y   Keep summary",       (dx0+34,   dy0+dh-34), F, 0.7, (255,255,255), 1, AA)
    cv2.rectangle(canvas, (dx0+dw-210,dy0+dh-64), (dx0+dw-18,dy0+dh-18), (170,50,50), -1)
    cv2.putText(canvas, "N   Dismiss",            (dx0+dw-200,dy0+dh-34), F, 0.7, (255,255,255), 1, AA)

# ── Calibration config ────────────────────────────────────────
CALIB_PTS = [
    # 4 corner anchors — guardrails outside the reading column
    (0.15, 0.15), (0.85, 0.15),
    (0.15, 0.85), (0.85, 0.85),
    # 3×3 dense grid in the PDF reading column (x: 35–65 %, y: 15–85 %)
    (0.35, 0.15), (0.50, 0.15), (0.65, 0.15),
    (0.35, 0.50), (0.50, 0.50), (0.65, 0.50),
    (0.35, 0.85), (0.50, 0.85), (0.65, 0.85),
]
# Drift correction points concentrated in the reading zone
DRIFT_PTS = [
    (0.50, 0.50),
    (0.35, 0.15), (0.65, 0.15),
    (0.35, 0.85), (0.65, 0.85),
]
SAMPLES_N = 20
DRIFT_N   = 10

# ── Main ──────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf',    required=True)
    ap.add_argument('--dwell',  type=float, default=5.)
    ap.add_argument('--zoom',   type=float, default=None,
                    help='PDF render zoom (default: auto-fit screen width)')
    ap.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    args = ap.parse_args()

    if not Path(args.pdf).exists():
        print(f"[ERROR] PDF not found: {args.pdf}"); sys.exit(1)

    device = torch.device(args.device)
    print("Loading ResNet50 (eye-crop mode) — ~100 MB download on first run...")
    gaze   = GazeEstimator(device)
    pdf    = PDFDocument(args.pdf, zoom=args.zoom); page = 0
    kalman  = KalmanGaze(pn=1e-3, mn=3e-2)
    hov_buf = deque(maxlen=20)               # paragraph vote buffer
    gx = SCREEN_W//2; gy = SCREEN_H//2
    scroll_y = 0; max_scroll = 0
    SCROLL_STEP = 120
    UP_KEY = 2490368; DOWN_KEY = 2621440    # Windows OpenCV arrow key codes

    ci = 0; cf = []; ct = []
    calib_done = threading.Event()
    samp = False; sbuf = []; training = [False]

    dfixes = []; dm = False; di = 0; ds = False; dbuf = []

    def add_drift(feat, tx, ty):
        dfixes.append((feat.copy(), [tx, ty]))
        if len(dfixes) > 20: dfixes.pop(0)
        fit_calibration(cf + [d[0] for d in dfixes], ct + [d[1] for d in dfixes])
        print(f"[Drift] {len(dfixes)} fix(es).")

    stuck       = StuckDetector(dwell=args.dwell, cooldown=20.)
    dlg         = False; dlg_para = None; dlg_summary = {'text': None}

    vs = WebcamStream().start(); time.sleep(0.5)
    print(f"PDF: {args.pdf} ({pdf.n} pages)  dwell={args.dwell}s")
    print("SPACE=confirm | n=next | p=prev | d=drift | r=recal | q=quit")
    print("Cyan circles = detected eye centres.  Green dot = tracking OK.")

    cv2.namedWindow('PDF Reader', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('PDF Reader', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.namedWindow('Camera Feed')

    last_fid = -1
    cam = np.zeros((360, 640, 3), dtype=np.uint8)

    while True:
        fid    = vs.fid
        canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

        if fid > last_fid:
            raw = vs.read()
            if raw is None: continue
            gaze.submit(raw)            # unflipped → estimator
            cam      = cv2.flip(raw, 1) # flipped  → selfie display
            last_fid = fid

        features, eye_pts, detected = gaze.read()

        if detected:
            if calib_done.is_set() and features is not None:
                sx, sy = predict_gaze(features)
                kx, ky = kalman.update(int(sx*SCREEN_W), int(sy*SCREEN_H))
                gx = int(np.clip(kx, 0, SCREEN_W-1))
                gy = int(np.clip(ky, 0, SCREEN_H-1))
            if samp and features is not None: sbuf.append(features.tolist())
            if ds   and features is not None: dbuf.append(features.tolist())

            # Draw mirrored eye circles on selfie cam feed
            if eye_pts:
                iw = cam.shape[1]
                for (ex, ey) in eye_pts:
                    cv2.circle(cam, (iw - ex, ey), 10, (0, 215, 255), 2)

        cv2.circle(cam, (20,20), 10, (0,220,80) if detected else (0,60,200), -1)

        # ── CALIBRATION ──────────────────────────────────────────
        if not calib_done.is_set():
            canvas[:] = 18
            if training[0]:
                cv2.putText(canvas, "Training model, please wait...",
                    (SCREEN_W//2-280, SCREEN_H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100,200,255), 2)
            else:
                tx = int(CALIB_PTS[ci][0]*SCREEN_W)
                ty = int(CALIB_PTS[ci][1]*SCREEN_H)
                if samp:
                    draw_dot(canvas, tx, ty, len(sbuf)/SAMPLES_N,
                             f"Hold still... {len(sbuf)}/{SAMPLES_N}")
                    if len(sbuf) >= SAMPLES_N:
                        avg = np.median(np.array(sbuf), axis=0)
                        cf.append(avg); ct.append(list(CALIB_PTS[ci]))
                        print(f"Point {ci+1}/{len(CALIB_PTS)} captured")
                        sbuf.clear(); samp = False; ci += 1
                        if ci == len(CALIB_PTS):
                            training[0] = True
                            def _train():
                                fit_calibration(cf, ct)
                                kalman.reset(); calib_done.set()
                                training[0] = False
                                print("Calibration complete!")
                            threading.Thread(target=_train, daemon=True).start()
                else:
                    p = int(10 + 6*abs(np.sin(time.time()*3)))
                    cv2.circle(canvas, (tx,ty), p+6, (255,255,255), 2)
                    cv2.circle(canvas, (tx,ty), p,   (0,0,220),     -1)
                    txt = f"Look at dot ({ci+1}/{len(CALIB_PTS)}) — eyes only, head still, SPACE"
                    tw  = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)[0][0]
                    cv2.putText(canvas, txt, ((SCREEN_W-tw)//2, SCREEN_H//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200,200,200), 2)

                for i, (px,py) in enumerate(CALIB_PTS):
                    col = (0,200,0) if i<ci else ((255,255,255) if i==ci else (80,80,80))
                    sz  = 8         if i<ci else (5             if i==ci else 6)
                    cv2.circle(canvas, (int(px*SCREEN_W), int(py*SCREEN_H)), sz, col, -1)

        # ── READING ───────────────────────────────────────────────
        else:
            pg_img, paras = pdf.get_page(page)
            ph, pw = pg_img.shape[:2]
            max_scroll = max(0, ph - SCREEN_H)
            scroll_y   = min(scroll_y, max_scroll)
            ox   = max(0, (SCREEN_W - pw) // 2)
            xe   = min(ox + pw, SCREEN_W)
            sh   = min(SCREEN_H, ph - scroll_y)     # visible slice height
            canvas[0:sh, ox:xe] = pg_img[scroll_y:scroll_y + sh, :xe - ox]

            # Gaze in page coordinates: x relative to page left, y with scroll offset
            gpx = gx - ox; gpy = gy + scroll_y; hov = None
            if not dlg and paras and pw > 0:
                ax0 = min(p["bbox"][0] for p in paras)
                ax1 = max(p["bbox"][2] for p in paras)
                hm  = (ax1-ax0)*0.35
                if ax0-hm <= gpx <= ax1+hm:
                    bd, bi = float('inf'), None
                    for pi, para in enumerate(paras):
                        bx0,by0,bx1,by1 = para["bbox"]
                        vd = 0 if by0<=gpy<=by1 else min(abs(gpy-by0), abs(gpy-by1))
                        if vd < 60 and vd < bd: bd=vd; bi=pi
                    hov = bi

            # Majority-vote over last 20 frames — one bad frame can't flip the paragraph
            hov_buf.append(hov)
            counts = {}
            for v in hov_buf:
                if v is not None: counts[v] = counts.get(v, 0) + 1
            nones = hov_buf.count(None)
            stable_hov = (max(counts, key=counts.get)
                          if counts and nones < len(hov_buf) // 2 else None)

            if not dlg:
                fired = stuck.update(stable_hov)
                if fired is not None and paras:
                    dlg = True; dlg_para = paras[fired]; dlg_summary['text'] = None
                    def _summ(text, box): box['text'] = summarize(text)
                    threading.Thread(target=_summ,
                                     args=(dlg_para["text"], dlg_summary),
                                     daemon=True).start()

            for pi, para in enumerate(paras):
                if pi == stable_hov:
                    bx0,by0,bx1,by1 = para["bbox"]
                    sy0 = by0 - scroll_y; sy1 = by1 - scroll_y
                    if sy1 > 0 and sy0 < SCREEN_H:
                        cv2.rectangle(canvas, (ox+bx0, sy0), (ox+bx1, sy1), (255,220,50), 2)

            if not dlg:
                cv2.line(canvas,   (gx-14,gy),  (gx+14,gy),  (0,255,180), 1)
                cv2.line(canvas,   (gx,gy-14),  (gx,gy+14),  (0,255,180), 1)
                cv2.circle(canvas, (gx,gy), 3,  (0,255,180), -1)

            scroll_pct = int(100 * scroll_y / max_scroll) if max_scroll > 0 else 0
            cv2.putText(canvas, f"Page {page+1}/{pdf.n}  [{scroll_pct}%]", (20,SCREEN_H-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120,120,120), 1)
            cv2.putText(canvas, "j/k=scroll  n=next  p=prev  d=drift  r=recal  q=quit",
                        (SCREEN_W-530,SCREEN_H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)

            if dm:
                ddx = int(DRIFT_PTS[di][0]*SCREEN_W)
                ddy = int(DRIFT_PTS[di][1]*SCREEN_H)
                cv2.putText(canvas, f"Drift ({di+1}/{len(DRIFT_PTS)}) — SPACE",
                            (20,SCREEN_H-50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,220,50), 2)
                if ds:
                    draw_dot(canvas, ddx, ddy, len(dbuf)/DRIFT_N,
                             f"Hold still... {len(dbuf)}/{DRIFT_N}")
                    if len(dbuf) >= DRIFT_N:
                        avg = np.median(np.array(dbuf), axis=0)
                        add_drift(avg, *DRIFT_PTS[di])
                        dbuf.clear(); ds=False; di+=1
                        if di >= len(DRIFT_PTS):
                            dm=False; di=0; kalman.reset(); stuck.reset()
                            print("Drift complete.")
                else:
                    pp = int(8+5*abs(np.sin(time.time()*3)))
                    cv2.circle(canvas, (ddx,ddy), pp+5, (255,220,50), 2)
                    cv2.circle(canvas, (ddx,ddy), pp,   (200,160,0),  -1)

            if dlg: draw_dialog(canvas, dlg_summary)

        cv2.imshow('PDF Reader',  canvas)
        cv2.imshow('Camera Feed', cam)
        key = cv2.waitKey(1)
        key_char = key & 0xFF

        if key_char == ord('q'): break
        elif key == UP_KEY   or key_char == ord('k'): scroll_y = max(0, scroll_y - SCROLL_STEP)
        elif key == DOWN_KEY or key_char == ord('j'): scroll_y = min(max_scroll, scroll_y + SCROLL_STEP)
        elif key_char in (ord('y'),ord('Y')) and dlg: dlg=False; dlg_para=None
        elif key_char in (ord('n'),ord('N')) and dlg:
            dlg=False; dlg_para=None; dlg_summary['text']=None
        elif not dlg:
            if key_char == ord('n') and calib_done.is_set():
                page=min(page+1,pdf.n-1); scroll_y=0; stuck.reset(); hov_buf.clear(); print(f"Page {page+1}/{pdf.n}")
            elif key_char == ord('p') and calib_done.is_set():
                page=max(page-1,0); scroll_y=0; stuck.reset(); hov_buf.clear(); print(f"Page {page+1}/{pdf.n}")
            elif key_char == ord('r'):
                ci=0; cf.clear(); ct.clear()
                calib_done.clear(); samp=False; sbuf.clear(); training[0]=False
                dm=False; di=0; ds=False; dbuf.clear(); dfixes.clear()
                gx=SCREEN_W//2; gy=SCREEN_H//2; kalman.reset(); stuck.reset(); hov_buf.clear()
                cv2.setWindowProperty('PDF Reader', cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
                print("Recalibrating...")
            elif key_char == ord('d') and calib_done.is_set() and not dm:
                dm=True; di=0; ds=False; dbuf.clear()
                print("Drift correction — look at each dot and press SPACE.")
            elif key_char == ord(' '):
                if not calib_done.is_set() and not samp and not training[0]:
                    samp=True; sbuf.clear(); print(f"Sampling point {ci+1}...")
                elif calib_done.is_set() and dm and not ds:
                    ds=True; dbuf.clear(); print(f"Sampling drift point {di+1}...")

    gaze.stop(); vs.stop(); pdf.close(); cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
