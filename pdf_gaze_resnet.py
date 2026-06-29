"""
PDF Gaze Reader
===============
PDF reading  : PyMuPDF, paragraph detection, stuck detector, Gemini summary dialog
Gaze tracking: imported from gaze_core.py (RetinaFace + ResNet50 eye crops)
"""

import cv2, time, numpy as np, threading, os, sys, argparse
from collections import deque
from pathlib import Path

try:
    import fitz
except ImportError:
    print("[ERROR] pip install pymupdf"); sys.exit(1)

from gaze_core import GazeReader, CALIB_PTS, DRIFT_PTS

SCREEN_W = 1920
SCREEN_H = 1080

# ── PDF ───────────────────────────────────────────────────────
class PDFDocument:
    def __init__(self, path, zoom=None):
        self.doc = fitz.open(path)
        self.n = len(self.doc)
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
F  = cv2.FONT_HERSHEY_DUPLEX

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

    reader = GazeReader(screen_w=SCREEN_W, screen_h=SCREEN_H, device=args.device)
    reader.start()

    pdf     = PDFDocument(args.pdf, zoom=args.zoom); page = 0
    hov_buf = deque(maxlen=20)
    scroll_y = 0; max_scroll = 0
    SCROLL_STEP = 120
    UP_KEY = 2490368; DOWN_KEY = 2621440

    ci = 0; training = [False]
    dm = False; di = 0

    stuck      = StuckDetector(dwell=args.dwell, cooldown=20.)
    dlg        = False; dlg_para = None; dlg_summary = {'text': None}

    print(f"PDF: {args.pdf} ({pdf.n} pages)  dwell={args.dwell}s")
    print("SPACE=confirm | j/k=scroll | n=next | p=prev | d=drift | r=recal | q=quit")

    cv2.namedWindow('PDF Reader', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('PDF Reader', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.namedWindow('Camera Feed')

    while True:
        canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        reader.update()
        gx, gy = reader.get_gaze()
        cam    = reader.get_display_frame()

        # ── CALIBRATION ──────────────────────────────────────────
        if not reader.is_calibrated:
            canvas[:] = 18
            if training[0]:
                cv2.putText(canvas, "Training model, please wait...",
                    (SCREEN_W//2-280, SCREEN_H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100,200,255), 2)
            else:
                tx = int(CALIB_PTS[ci][0]*SCREEN_W)
                ty = int(CALIB_PTS[ci][1]*SCREEN_H)
                if reader._collecting:
                    prog     = reader.calibration_progress
                    n_so_far = int(prog * reader.CAL_SAMPLES)
                    draw_dot(canvas, tx, ty, prog,
                             f"Hold still... {n_so_far}/{reader.CAL_SAMPLES}")
                    if reader.calibration_point_ready:
                        reader.commit_calibration_point()
                        print(f"Point {ci+1}/{len(CALIB_PTS)} captured")
                        ci += 1
                        if ci == len(CALIB_PTS):
                            training[0] = True
                            def _train():
                                reader.fit()
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
            ox  = max(0, (SCREEN_W - pw) // 2)
            xe  = min(ox + pw, SCREEN_W)
            sh  = min(SCREEN_H, ph - scroll_y)
            canvas[0:sh, ox:xe] = pg_img[scroll_y:scroll_y + sh, :xe - ox]

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
                if reader._is_drift and reader._collecting:
                    prog = reader.calibration_progress
                    draw_dot(canvas, ddx, ddy, prog,
                             f"Hold still... {int(prog*reader.DRIFT_SAMPLES)}/{reader.DRIFT_SAMPLES}")
                    if reader.drift_point_ready:
                        reader.commit_drift_point()
                        di += 1
                        if di >= len(DRIFT_PTS):
                            dm=False; di=0; stuck.reset()
                            print("Drift complete.")
                else:
                    pp = int(8+5*abs(np.sin(time.time()*3)))
                    cv2.circle(canvas, (ddx,ddy), pp+5, (255,220,50), 2)
                    cv2.circle(canvas, (ddx,ddy), pp,   (200,160,0),  -1)

            if dlg: draw_dialog(canvas, dlg_summary)

        cv2.imshow('PDF Reader',  canvas)
        cv2.imshow('Camera Feed', cam)
        key      = cv2.waitKey(1)
        key_char = key & 0xFF

        if key_char == ord('q'): break
        elif key == UP_KEY   or key_char == ord('k'): scroll_y = max(0, scroll_y - SCROLL_STEP)
        elif key == DOWN_KEY or key_char == ord('j'): scroll_y = min(max_scroll, scroll_y + SCROLL_STEP)
        elif key_char in (ord('y'),ord('Y')) and dlg: dlg=False; dlg_para=None
        elif key_char in (ord('n'),ord('N')) and dlg:
            dlg=False; dlg_para=None; dlg_summary['text']=None
        elif not dlg:
            if key_char == ord('n') and reader.is_calibrated:
                page=min(page+1,pdf.n-1); scroll_y=0; stuck.reset(); hov_buf.clear()
                print(f"Page {page+1}/{pdf.n}")
            elif key_char == ord('p') and reader.is_calibrated:
                page=max(page-1,0); scroll_y=0; stuck.reset(); hov_buf.clear()
                print(f"Page {page+1}/{pdf.n}")
            elif key_char == ord('r'):
                ci=0; training[0]=False; dm=False; di=0
                reader.reset_calibration()
                stuck.reset(); hov_buf.clear()
                cv2.setWindowProperty('PDF Reader', cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
                print("Recalibrating...")
            elif key_char == ord('d') and reader.is_calibrated and not dm:
                dm=True; di=0
                print("Drift correction — look at each dot and press SPACE.")
            elif key_char == ord(' '):
                if not reader.is_calibrated and not reader._collecting and not training[0]:
                    reader.begin_calibration_point(*CALIB_PTS[ci])
                    print(f"Sampling point {ci+1}...")
                elif reader.is_calibrated and dm and not (reader._is_drift and reader._collecting):
                    reader.begin_drift_point(*DRIFT_PTS[di])
                    print(f"Sampling drift point {di+1}...")

    reader.stop(); pdf.close(); cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
