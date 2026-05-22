"""
PDF Gaze Reader
===============
Eye tracking: v1 (GBR + Kalman + drift correction)
PDF reading:  PyMuPDF, paragraph detection, stuck detector, summary dialog

Controls
--------
  SPACE   confirm calibration dot / drift point
  n       next page
  p       previous page
  d       drift correction (after calibration)
  r       recalibrate
  q       quit

  When stuck dialog appears:
  Y       keep summary
  N       dismiss
"""

import cv2, time, numpy as np, mediapipe as mp, threading, os, sys, argparse
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor

try:
    import fitz
except ImportError:
    print("[ERROR] pip install pymupdf"); sys.exit(1)

# =============================================================
# Screen
# =============================================================
SCREEN_W = 1920
SCREEN_H = 1080

# =============================================================
# Landmark indices
# =============================================================
IRIS_LEFT        = [474, 475, 476, 477]
IRIS_RIGHT       = [469, 470, 471, 472]
EYE_LEFT_OUTER   = 33;  EYE_LEFT_INNER  = 133
EYE_LEFT_TOP     = [159,160,161]; EYE_LEFT_BOTTOM  = [145,144,163]
EYE_RIGHT_OUTER  = 362; EYE_RIGHT_INNER = 263
EYE_RIGHT_TOP    = [386,387,388]; EYE_RIGHT_BOTTOM = [374,373,390]
_HEAD_IDX = [1,152,33,263,61,291]
_HEAD_3D  = np.array([
    [0.,0.,0.],[0.,-330.,-65.],[-225.,170.,-135.],
    [225.,170.,-135.],[-150.,-150.,-125.],[150.,-150.,-125.]], dtype=np.float64)

# =============================================================
# Kalman  (v1)
# =============================================================
class KalmanGaze:
    def __init__(self, pn=1e-4, mn=8e-2, dt=1/30):
        self.kf = cv2.KalmanFilter(4,2)
        self.kf.transitionMatrix    = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],dtype=np.float32)
        self.kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]],dtype=np.float32)
        self.kf.processNoiseCov     = np.eye(4,dtype=np.float32)*pn
        self.kf.measurementNoiseCov = np.eye(2,dtype=np.float32)*mn
        self.kf.errorCovPost        = np.eye(4,dtype=np.float32)
        self._ok = False
    def update(self,x,y):
        if not self._ok:
            self.kf.statePre = self.kf.statePost = np.array([[x],[y],[0],[0]],dtype=np.float32)
            self._ok = True; return x,y
        self.kf.predict()
        c = self.kf.correct(np.array([[x],[y]],dtype=np.float32))
        return float(c[0,0]), float(c[1,0])
    def reset(self): self._ok = False

# =============================================================
# Feature functions  (v1)
# =============================================================
def get_head_pose(lm, iw, ih):
    pts = np.array([(lm[i].x*iw, lm[i].y*ih) for i in _HEAD_IDX], dtype=np.float64)
    fl  = float(iw)
    cam = np.array([[fl,0,iw/2],[0,fl,ih/2],[0,0,1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_HEAD_3D, pts, cam, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return 0.,0.,0.,np.eye(3)
    R,_ = cv2.Rodrigues(rvec)
    sy  = np.sqrt(R[0,0]**2+R[1,0]**2)
    if sy>1e-6:
        yaw=np.degrees(np.arctan2(-R[2,0],sy)); pitch=np.degrees(np.arctan2(R[2,1],R[2,2])); roll=np.degrees(np.arctan2(R[1,0],R[0,0]))
    else:
        pitch=np.degrees(np.arctan2(-R[2,0],sy)); yaw=roll=0.
    return yaw,pitch,roll,R

def get_face_scale(lm):
    return float(np.hypot(lm[263].x-lm[33].x, lm[263].y-lm[33].y))

def get_compensated_gaze(lm_px, R, iw, ih):
    nose  = np.array(lm_px[1], dtype=np.float64)
    def ic(idx): return np.mean([np.array(lm_px[i],dtype=np.float64) for i in idx],axis=0)
    fw    = np.hypot(lm_px[263][0]-lm_px[33][0], lm_px[263][1]-lm_px[33][1])+1e-6
    def comp(pt):
        v = np.array([(pt[0]-nose[0])/fw,(pt[1]-nose[1])/fw,0.]); vc=R.T@v
        return float(vc[0]),float(vc[1])
    lx,ly = comp(ic(IRIS_LEFT)); rx,ry = comp(ic(IRIS_RIGHT))
    return lx,ly,rx,ry

def get_eye_ear(lm_px, outer, inner, top_ids, bottom_ids):
    lx,ly=lm_px[outer]; rx,ry=lm_px[inner]
    ew=np.hypot(rx-lx,ry-ly)
    ty=sum(lm_px[i][1] for i in top_ids)/len(top_ids)
    by=sum(lm_px[i][1] for i in bottom_ids)/len(bottom_ids)
    return abs(by-ty)/(ew+1e-6)

def build_feature(lx,ly,rx,ry,yaw,pitch,fs):
    yn=yaw/30.; pn=pitch/20.
    return np.array([lx,ly,rx,ry,yn,pn,fs,lx*yn,rx*yn,ly*pn,ry*pn],dtype=np.float64)

# =============================================================
# ML  (v1)
# =============================================================
def _make_model():
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2,include_bias=False),
        GradientBoostingRegressor(n_estimators=100,max_depth=3,learning_rate=0.1,random_state=0))

model_x = _make_model()
model_y = _make_model()

def fit_calibration(features, targets, weights=None):
    X=np.array(features); Y=np.array(targets); sw=None
    if weights and len(weights)==len(X):
        W=np.clip(np.array(weights),1e-6,None); sw=W/W.max()
    model_x.fit(X,Y[:,0],gradientboostingregressor__sample_weight=sw)
    model_y.fit(X,Y[:,1],gradientboostingregressor__sample_weight=sw)

def predict(feat):
    p=feat.reshape(1,-1)
    return float(model_x.predict(p)[0]), float(model_y.predict(p)[0])

# =============================================================
# Range Stretcher  (v1)
# =============================================================
class RangeStretcher:
    def __init__(self,margin=0.02):
        self.margin=margin; self.fitted=False
        self.x0=self.x1=self.y0=self.y1=None
    def fit(self,features,targets):
        X=np.array(features)
        rx=model_x.predict(X); ry=model_y.predict(X)
        self.x0,self.x1=rx.min(),rx.max(); self.y0,self.y1=ry.min(),ry.max()
        sx=self.x1-self.x0; sy=self.y1-self.y0
        self.x0+=sx*self.margin; self.x1-=sx*self.margin
        self.y0+=sy*self.margin; self.y1-=sy*self.margin
        self.fitted=True
        print(f"[Stretcher] X=[{self.x0:.3f},{self.x1:.3f}] Y=[{self.y0:.3f},{self.y1:.3f}]")
    def apply(self,rx,ry):
        if not self.fitted: return float(np.clip(rx,0,1)),float(np.clip(ry,0,1))
        return float(np.clip((rx-self.x0)/max(self.x1-self.x0,1e-6),0,1)), \
               float(np.clip((ry-self.y0)/max(self.y1-self.y0,1e-6),0,1))

stretcher = RangeStretcher()

def predict_stretched(feat):
    rx,ry=predict(feat); return stretcher.apply(rx,ry)

# =============================================================
# Session history — persists calibration data across runs
# =============================================================
HISTORY_FILE = 'calibration_history.npz'
MAX_HISTORY_SESSIONS = 20   # keep at most this many sessions

def save_session(features, targets):
    """Append this session's data to the history file."""
    features = np.array(features, dtype=np.float64)
    targets  = np.array(targets,  dtype=np.float64)

    fs_mean = float(features[:, 6].mean())
    if fs_mean > 1e-6:
        features = features.copy()
        features[:, 6] /= fs_mean

    if Path(HISTORY_FILE).exists():
        d        = np.load(HISTORY_FILE, allow_pickle=True)
        old_f    = d['features']
        old_t    = d['targets']
        old_sid  = d['session_id'].astype(np.int32)
        old_fs   = list(d['fs_means'])
        next_id  = int(old_sid[-1]) + 1
        # split into per-session lists for trimming
        sessions = list(np.unique(old_sid))
        sess_f   = [old_f[old_sid == s] for s in sessions]
        sess_t   = [old_t[old_sid == s] for s in sessions]
        sess_s   = [np.full(int((old_sid==s).sum()), int(s), dtype=np.int32) for s in sessions]
        all_fs   = old_fs
    else:
        sess_f, sess_t, sess_s, all_fs = [], [], [], []
        next_id  = 1

    sess_f.append(features)
    sess_t.append(targets)
    sess_s.append(np.full(len(features), next_id, dtype=np.int32))
    all_fs.append(fs_mean)

    if len(sess_f) > MAX_HISTORY_SESSIONS:
        sess_f = sess_f[-MAX_HISTORY_SESSIONS:]
        sess_t = sess_t[-MAX_HISTORY_SESSIONS:]
        sess_s = sess_s[-MAX_HISTORY_SESSIONS:]
        all_fs = all_fs[-MAX_HISTORY_SESSIONS:]

    np.savez_compressed(HISTORY_FILE,
        features   = np.vstack(sess_f),
        targets    = np.vstack(sess_t),
        session_id = np.concatenate(sess_s).astype(np.int32),
        fs_means   = np.array(all_fs, dtype=np.float64))

    total = sum(len(f) for f in sess_f)
    print(f"[History] Saved session {next_id} — {len(features)} pts. "
          f"Total history: {total} pts across {len(sess_f)} sessions.")


def load_history(current_features, current_targets, current_weights):
    """
    Merge history with current session data.
    Returns (features, targets, weights) ready for fit_calibration().
    Recent sessions get higher weight; current session highest.
    """
    history_file = HISTORY_FILE
    if not Path(history_file).exists():
        return current_features, current_targets, current_weights

    d          = np.load(history_file, allow_pickle=True)
    hist_f     = d['features']
    hist_t     = d['targets']
    hist_s     = d['session_id']
    sessions   = np.unique(hist_s)
    n_sessions = len(sessions)

    # Weight: oldest session = 0.3, newest history = 0.7, current = 1.0
    hist_w = []
    for sid in hist_s:
        rank      = np.where(sessions == sid)[0][0]           # 0 = oldest
        w         = 0.3 + 0.4 * (rank / max(n_sessions-1, 1)) # 0.3 → 0.7
        hist_w.append(w)

    all_f = list(hist_f) + list(current_features)
    all_t = list(hist_t) + list(current_targets)
    all_w = hist_w       + [1.0] * len(current_features)   # current = weight 1.0

    # Scale current weights by their confidence values
    if current_weights:
        cw_arr = np.array(current_weights); cw_arr = cw_arr / cw_arr.max()
        all_w[-len(current_features):] = cw_arr.tolist()

    print(f"[History] Using {len(hist_f)} history pts + {len(current_features)} current pts "
          f"({n_sessions} past sessions).")
    return all_f, all_t, all_w


# =============================================================
# Outlier rejection  (v1)
# =============================================================
def reject_outliers(samples,k=2.0):
    arr=np.array(samples); med=np.median(arr,axis=0)
    mad=np.median(np.abs(arr-med),axis=0)+1e-9
    mask=np.all(np.abs(arr-med)<=k*mad,axis=1); good=arr[mask]
    return good if len(good)>=max(5,len(arr)//4) else arr

# =============================================================
# Camera thread
# =============================================================
class WebcamStream:
    def __init__(self,src=0,w=1280,h=720):
        self.cap=cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,w); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,h)
        _,self.frame=self.cap.read(); self.stopped=False; self.fid=0
    def start(self):
        threading.Thread(target=self._run,daemon=True).start(); return self
    def _run(self):
        while not self.stopped:
            ok,f=self.cap.read()
            if ok: self.frame=f; self.fid+=1
    def read(self): return self.frame
    def stop(self): self.stopped=True; self.cap.release()

# =============================================================
# PDF
# =============================================================
class PDFDocument:
    def __init__(self,path,zoom=1.5):
        self.doc=fitz.open(path); self.zoom=zoom; self.n=len(self.doc); self._cache={}
    def get_page(self,num):
        if num in self._cache: return self._cache[num]
        page=self.doc[num]
        mat=fitz.Matrix(self.zoom,self.zoom)
        pix=page.get_pixmap(matrix=mat,alpha=False)
        img=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.h,pix.w,3)
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        ph=page.rect.height
        paras=[]
        for b in page.get_text("dict",flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
            if b["type"]!=0: continue
            x0,y0,x1,y1=[c*self.zoom for c in b["bbox"]]
            text=" ".join(s["text"] for l in b.get("lines",[]) for s in l.get("spans",[])).strip()
            if len(text)<80: continue
            if b["bbox"][1]<ph*0.12: continue
            lines=b.get("lines",[])
            if len(lines)<=2:
                avg=np.mean([s.get("size",10) for l in lines for s in l.get("spans",[])] or [10])
                if avg>13: continue
            paras.append({"bbox":(int(x0),int(y0),int(x1),int(y1)),"text":text})
        self._cache[num]=(img,paras); return img,paras
    def close(self): self.doc.close()

# =============================================================
# Stuck detector
# =============================================================
class StuckDetector:
    def __init__(self,dwell=5.,cooldown=20.):
        self.dwell=dwell; self.cooldown=cooldown
        self._cur=None; self._t0=None; self._fired={}; self._fv=False
    def update(self,idx):
        now=time.time()
        if idx!=self._cur:
            self._cur=idx; self._t0=now if idx is not None else None; self._fv=False; return None
        if idx is None or self._fv: return None
        if now-self._t0>=self.dwell and now-self._fired.get(idx,0)>=self.cooldown:
            self._fired[idx]=now; self._fv=True; return idx
        return None
    def reset(self): self._cur=None; self._t0=None; self._fv=False; self._fired.clear()

# =============================================================
# Summarizer
# =============================================================
def summarize(text):
    key=os.environ.get("ANTHROPIC_API_KEY","")
    if key:
        try:
            import anthropic
            c=anthropic.Anthropic(api_key=key)
            m=c.messages.create(model="claude-sonnet-4-20250514",max_tokens=300,
                messages=[{"role":"user","content":
                    "Summarize in 2-3 simple sentences for a student struggling with this:\n\n"+text}])
            return m.content[0].text.strip()
        except Exception as e: return f"[API error: {e}]"
    return "[Set ANTHROPIC_API_KEY for AI summary]\n\n"+text[:400]

# =============================================================
# UI helpers
# =============================================================
def draw_dot(canvas,tx,ty,prog,label):
    cv2.ellipse(canvas,(tx,ty),(22,22),-90,0,int(360*prog),(0,255,100),3)
    cv2.circle(canvas,(tx,ty),12,(0,200,80),-1)
    tw=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.9,2)[0][0]
    cv2.putText(canvas,label,((SCREEN_W-tw)//2,SCREEN_H//2),cv2.FONT_HERSHEY_SIMPLEX,0.9,(200,200,200),2)

def wrap_text(text,n=82):
    words=text.split(); lines=[]; line=""
    for w in words:
        if len(line)+len(w)+1>n:
            if line: lines.append(line)
            line=w
        else: line=(line+" "+w).strip()
    if line: lines.append(line)
    return lines


# =============================================================
# Calibration points  (PDF-optimised)
# =============================================================
CALIB_PTS = [
    (0.50,0.50),(0.50,0.10),(0.50,0.90),(0.22,0.50),(0.78,0.50),
    (0.22,0.15),(0.78,0.15),(0.22,0.85),(0.78,0.85),
    (0.50,0.25),(0.50,0.75),(0.35,0.50),(0.65,0.50),
    (0.35,0.25),(0.65,0.25),(0.35,0.75),(0.65,0.75),
    (0.10,0.30),(0.90,0.30),(0.10,0.70),(0.90,0.70),
]
DRIFT_PTS = [(0.50,0.50),(0.22,0.15),(0.78,0.15),(0.22,0.85),(0.78,0.85)]
SAMPLES_N  = 15
DRIFT_N    = 10

# =============================================================
# MediaPipe shared state  (dict avoids global keyword issues)
# =============================================================
_mp = {'lm': None, 'ready': False}
_mp_lock = threading.Lock()

def _mp_cb(result, output_image, timestamp_ms):
    with _mp_lock:
        _mp['lm']    = result.face_landmarks[0] if result.face_landmarks else None
        _mp['ready'] = True

# =============================================================
# Main
# =============================================================
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pdf',required=True)
    ap.add_argument('--dwell',type=float,default=5.)
    ap.add_argument('--zoom',type=float,default=1.5)
    ap.add_argument('--model',default='face_landmarker.task')
    args=ap.parse_args()

    if not Path(args.pdf).exists(): print(f"[ERROR] PDF not found: {args.pdf}"); sys.exit(1)
    if not Path(args.model).exists(): print("[ERROR] face_landmarker.task not found"); sys.exit(1)



    # MediaPipe
    det=vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=args.model),
            num_faces=1, min_face_detection_confidence=0.5, min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.LIVE_STREAM, result_callback=_mp_cb))

    pdf=PDFDocument(args.pdf,zoom=args.zoom); page=0
    kalman=KalmanGaze()
    gx=SCREEN_W//2; gy=SCREEN_H//2
    feat=None; stale=0; MAX_STALE=15
    lrx=lry=rrx=rry=yaw_v=pitch_v=fs_v=0.

    # calibration
    ci=0; cf=[]; ct=[]; cw=[]
    calib_done = threading.Event()
    samp=False; sbuf=[]; _tr={"v":False}  # dict so nested fn can mutate it

    # drift
    dfixes=[]; dm=False; di=0; ds=False; dbuf=[]

    def add_drift(f,tx,ty):
        dfixes.append((f.copy(),[tx,ty]))
        if len(dfixes)>30: dfixes.pop(0)
        af=cf+[d[0] for d in dfixes]; at=ct+[d[1] for d in dfixes]
        bw=max(cw,default=1.); aw=cw+[3.*bw]*len(dfixes)
        fit_calibration(af,at,aw); stretcher.fit(af,at)
        print(f"[Drift] {len(dfixes)} fix(es).")

    # stuck + dialog
    stuck=StuckDetector(dwell=args.dwell,cooldown=20.)
    dlg=False; dlg_para=None
    dlg_summary={'text':None}   # dict so thread can write to it safely

    vs=WebcamStream().start(); time.sleep(0.8)
    print(f"PDF: {args.pdf} ({pdf.n} pages) dwell={args.dwell}s")
    print("SPACE=confirm | n=next | p=prev | d=drift | r=recal | q=quit")

    cv2.namedWindow('PDF Reader',cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('PDF Reader',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.namedWindow('Camera Feed')

    last_fid=-1; disp=None; lm_cur=None

    while True:
        fid=vs.fid
        canvas=np.zeros((SCREEN_H,SCREEN_W,3),dtype=np.uint8)

        # camera
        if fid>last_fid:
            raw=vs.read()
            if raw is None: continue
            disp=cv2.flip(raw,1)
            det.detect_async(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)),
                int(time.time()*1000))
            last_fid=fid
        if disp is None: continue
        cam=disp.copy(); ih,iw=cam.shape[:2]

        # MediaPipe result
        with _mp_lock:
            new_lm=_mp['ready']
            if new_lm:
                lm_cur=_mp['lm']; _mp['ready']=False

        # feature extraction
        if new_lm and lm_cur:
            lm_px=[(lm.x*iw,lm.y*ih) for lm in lm_cur]
            yaw,pitch,roll,R=get_head_pose(lm_cur,iw,ih)
            fs=get_face_scale(lm_cur)
            el=get_eye_ear(lm_px,EYE_LEFT_OUTER,EYE_LEFT_INNER,EYE_LEFT_TOP,EYE_LEFT_BOTTOM)
            er=get_eye_ear(lm_px,EYE_RIGHT_OUTER,EYE_RIGHT_INNER,EYE_RIGHT_TOP,EYE_RIGHT_BOTTOM)
            eyes_open=el>0.10 and er>0.10

            use=eyes_open or (feat is not None and stale<MAX_STALE)
            if use:
                if eyes_open:
                    lrx,lry,rrx,rry=get_compensated_gaze(lm_px,R,iw,ih)
                    yaw_v,pitch_v,fs_v=yaw,pitch,fs
                    feat=build_feature(lrx,lry,rrx,rry,yaw_v,pitch_v,fs_v); stale=0
                else:
                    stale+=1

                if calib_done.is_set() and feat is not None:
                    sx,sy=predict_stretched(feat)
                    kx,ky=kalman.update(int(sx*SCREEN_W),int(sy*SCREEN_H))
                    gx=int(np.clip(kx,0,SCREEN_W-1)); gy=int(np.clip(ky,0,SCREEN_H-1))

                if samp and eyes_open: sbuf.append(feat.tolist())
                if ds   and eyes_open: dbuf.append(feat.tolist())

            for ig in [IRIS_LEFT,IRIS_RIGHT]:
                pts=[lm_px[i] for i in ig]
                cv2.circle(cam,(int(sum(p[0] for p in pts)/4),int(sum(p[1] for p in pts)/4)),4,(0,215,255),-1)

        elif new_lm: stale=0

        # ── CALIBRATION ──────────────────────────────────────────
        if not calib_done.is_set():
            canvas[:]=18
            if _tr["v"]:
                cv2.putText(canvas,"Training model, please wait...",
                    (SCREEN_W//2-280,SCREEN_H//2),cv2.FONT_HERSHEY_SIMPLEX,1.2,(100,200,255),2)
            else:
                tx=int(CALIB_PTS[ci][0]*SCREEN_W); ty=int(CALIB_PTS[ci][1]*SCREEN_H)
                if samp:
                    draw_dot(canvas,tx,ty,len(sbuf)/SAMPLES_N,
                             f"Hold still... {len(sbuf)}/{SAMPLES_N}")
                    if len(sbuf)>=SAMPLES_N:
                        clean=reject_outliers(sbuf); avg=clean.mean(axis=0)
                        conf=1./(clean.var(axis=0).mean()+1e-6)
                        cf.append(avg); ct.append(list(CALIB_PTS[ci])); cw.append(conf)
                        print(f"Point {ci+1}/{len(CALIB_PTS)} captured ({len(clean)}/{len(sbuf)} kept)")
                        sbuf.clear(); samp=False; ci+=1
                        if ci==len(CALIB_PTS):
                            _tr["v"]=True
                            def _train():
                                # Merge with history from previous sessions
                                all_f, all_t, all_w = load_history(cf, ct, cw)
                                fit_calibration(all_f, all_t, all_w)
                                stretcher.fit(all_f, all_t)
                                # Save this session to history for future runs
                                save_session(cf, ct)
                                kalman.reset()
                                calib_done.set()
                                _tr["v"]=False
                                print("Calibration complete! n/p to navigate pages.")
                            threading.Thread(target=_train,daemon=True).start()
                else:
                    p=int(10+6*abs(np.sin(time.time()*3)))
                    cv2.circle(canvas,(tx,ty),p+6,(255,255,255),2)
                    cv2.circle(canvas,(tx,ty),p,(0,0,220),-1)
                    txt=f"Look at dot ({ci+1}/{len(CALIB_PTS)}) — head still, SPACE"
                    tw=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,1.1,2)[0][0]
                    cv2.putText(canvas,txt,((SCREEN_W-tw)//2,SCREEN_H//2),
                                cv2.FONT_HERSHEY_SIMPLEX,1.1,(200,200,200),2)
                for i,(px,py) in enumerate(CALIB_PTS):
                    col,sz=((0,200,0),8) if i<ci else (((255,255,255),5) if i==ci else ((80,80,80),6))
                    cv2.circle(canvas,(int(px*SCREEN_W),int(py*SCREEN_H)),sz,col,-1)

        # ── READING ───────────────────────────────────────────────
        else:
            pg_img,paras=pdf.get_page(page)
            ph,pw=pg_img.shape[:2]
            ox=max(0,(SCREEN_W-pw)//2); oy=max(0,(SCREEN_H-ph)//2)
            x1=min(ox+pw,SCREEN_W); y1=min(oy+ph,SCREEN_H)
            canvas[oy:y1,ox:x1]=pg_img[:y1-oy,:x1-ox]

            gpx=gx-ox; gpy=gy-oy
            hov=None
            if not dlg and paras and pw>0:
                ax0=min(p["bbox"][0] for p in paras); ax1=max(p["bbox"][2] for p in paras)
                hm=(ax1-ax0)*0.35
                if ax0-hm<=gpx<=ax1+hm:
                    bd,bi=float('inf'),None
                    for pi,para in enumerate(paras):
                        bx0,by0,bx1,by1=para["bbox"]
                        vd=0 if by0<=gpy<=by1 else min(abs(gpy-by0),abs(gpy-by1))
                        if vd<60 and vd<bd: bd=vd; bi=pi
                    hov=bi

            if not dlg:
                fired=stuck.update(hov)
                if fired is not None and paras:
                    dlg=True; dlg_para=paras[fired]; dlg_summary['text']=None
                    def _summ(text,box):
                        box['text']=summarize(text)
                    threading.Thread(target=_summ,args=(dlg_para["text"],dlg_summary),daemon=True).start()

            for pi,para in enumerate(paras):
                bx0,by0,bx1,by1=para["bbox"]
                if pi==hov:
                    cv2.rectangle(canvas,(ox+bx0,oy+by0),(ox+bx1,oy+by1),(255,220,50),2)

            if not dlg:
                cv2.line(canvas,(gx-14,gy),(gx+14,gy),(0,255,180),1)
                cv2.line(canvas,(gx,gy-14),(gx,gy+14),(0,255,180),1)
                cv2.circle(canvas,(gx,gy),3,(0,255,180),-1)

            cv2.putText(canvas,f"Page {page+1}/{pdf.n}",(20,SCREEN_H-15),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(120,120,120),1)
            cv2.putText(canvas,"n=next  p=prev  d=drift  r=recal  q=quit",
                        (SCREEN_W-460,SCREEN_H-15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,100,100),1)

            if dm:
                ddx=int(DRIFT_PTS[di][0]*SCREEN_W); ddy=int(DRIFT_PTS[di][1]*SCREEN_H)
                cv2.putText(canvas,f"Drift ({di+1}/{len(DRIFT_PTS)}) — SPACE",
                            (20,SCREEN_H-50),cv2.FONT_HERSHEY_SIMPLEX,0.85,(255,220,50),2)
                if ds:
                    draw_dot(canvas,ddx,ddy,len(dbuf)/DRIFT_N,f"Hold still... {len(dbuf)}/{DRIFT_N}")
                    if len(dbuf)>=DRIFT_N:
                        clean=reject_outliers(dbuf); avg=clean.mean(axis=0)
                        add_drift(avg,*DRIFT_PTS[di])
                        dbuf.clear(); ds=False; di+=1
                        if di>=len(DRIFT_PTS):
                            dm=False; di=0; kalman.reset(); stuck.reset()
                            print("Drift complete.")
                else:
                    pp=int(8+5*abs(np.sin(time.time()*3)))
                    cv2.circle(canvas,(ddx,ddy),pp+5,(255,220,50),2)
                    cv2.circle(canvas,(ddx,ddy),pp,(200,160,0),-1)

            if dlg:
                dim=canvas.copy(); canvas[:]=canvas//3
                cv2.addWeighted(dim,0.15,canvas,0.85,0,canvas)
                dw2,dh2=760,420; dx0=(SCREEN_W-dw2)//2; dy0=(SCREEN_H-dh2)//2
                cv2.rectangle(canvas,(dx0,dy0),(dx0+dw2,dy0+dh2),(30,30,40),-1)
                cv2.rectangle(canvas,(dx0,dy0),(dx0+dw2,dy0+dh2),(100,180,255),2)
                cv2.putText(canvas,"Looks like you might be stuck on this paragraph.",
                            (dx0+20,dy0+38),cv2.FONT_HERSHEY_SIMPLEX,0.72,(100,200,255),2)
                sumtext=dlg_summary['text']
                if sumtext is None:
                    cv2.putText(canvas,"Generating summary...",(dx0+20,dy0+100),
                                cv2.FONT_HERSHEY_SIMPLEX,0.65,(180,180,180),1)
                else:
                    for i,line in enumerate(wrap_text(sumtext)[:12]):
                        cv2.putText(canvas,line,(dx0+20,dy0+80+i*28),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(220,220,220),1)
                cv2.rectangle(canvas,(dx0+20,dy0+dh2-60),(dx0+200,dy0+dh2-20),(0,180,80),-1)
                cv2.putText(canvas,"Y  Keep summary",(dx0+30,dy0+dh2-34),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
                cv2.rectangle(canvas,(dx0+dw2-180,dy0+dh2-60),(dx0+dw2-20,dy0+dh2-20),(180,60,60),-1)
                cv2.putText(canvas,"N  Dismiss",(dx0+dw2-170,dy0+dh2-34),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        cv2.imshow('PDF Reader',canvas)
        cv2.imshow('Camera Feed',cam)
        key=cv2.waitKey(1)&0xFF

        if key==ord('q'): break
        elif key in (ord('y'),ord('Y')) and dlg: dlg=False; dlg_para=None
        elif key in (ord('n'),ord('N')) and dlg: dlg=False; dlg_para=None; dlg_summary['text']=None
        elif not dlg:
            if key==ord('n'):
                page=min(page+1,pdf.n-1); stuck.reset(); print(f"Page {page+1}/{pdf.n}")
            elif key==ord('p'):
                page=max(page-1,0); stuck.reset(); print(f"Page {page+1}/{pdf.n}")
            elif key==ord('r'):
                ci=0; cf.clear(); ct.clear(); cw.clear()
                calib_done.clear(); samp=False; sbuf.clear(); _tr["v"]=False
                dm=False; di=0; ds=False; dbuf.clear(); dfixes.clear()
                gx=SCREEN_W//2; gy=SCREEN_H//2; kalman.reset(); stuck.reset()
                cv2.setWindowProperty('PDF Reader',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                print("Recalibrating...")
            elif key==ord('d') and calib_done.is_set() and not dm:
                dm=True; di=0; ds=False; dbuf.clear()
                print("Drift correction — look at each dot and press SPACE.")
            elif key==ord(' '):
                if not calib_done.is_set() and not samp and not _tr["v"]:
                    samp=True; sbuf.clear(); print(f"Sampling point {ci+1}...")
                elif calib_done.is_set() and dm and not ds:
                    ds=True; dbuf.clear(); print(f"Sampling drift point {di+1}...")

    vs.stop(); det.close(); pdf.close(); cv2.destroyAllWindows()

if __name__=='__main__':
    main()
