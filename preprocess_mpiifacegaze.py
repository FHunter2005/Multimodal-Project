"""
MPIIFaceGaze → eye_follower_v3 feature extractor
==================================================

What this script does
---------------------
1. Walks the MPIIFaceGaze dataset (p00–p14).
2. For every image, runs MediaPipe FaceMesh to get the same 478 landmarks
   that eye_follower_v3.py uses at runtime.
3. Computes exactly the same build_feature() vector that v3 computes live.
4. Normalises the gaze target to [0,1] using each participant's screen size
   from their Calibration/screenSize.mat.
5. Saves everything to a single compressed .npz file:
       pretrain_features.npz
           features : float64 array (N, 18)  ← same dim as v3 build_feature
           targets  : float64 array (N,  2)  ← [gaze_x_norm, gaze_y_norm]
           participants : int array  (N,)    ← 0..14, useful for LOO-CV

Known dataset quirks (handled)
-------------------------------
- Participants 2, 7, 10 have a small fraction of gaze points outside the
  reported screen bounds.  Those rows are dropped (MPIIFaceGaze- clean).
- The face images have the background blacked out; MediaPipe still detects
  the face reliably on them.
- screenSize.mat uses MATLAB-style 1×1 struct arrays; read with scipy.io.

Usage
-----
    pip install mediapipe scipy opencv-python numpy tqdm
    python preprocess_mpiifacegaze.py --dataset /path/to/MPIIFaceGaze
                                      --output  pretrain_features.npz
                                      --max_per_participant 2000

Arguments
---------
--dataset   Root folder of MPIIFaceGaze (contains p00, p01, …, p14)
--output    Output .npz file path  (default: pretrain_features.npz)
--max_per_participant
            Subsample to at most N rows per participant (default: 2500).
            The full dataset has ~2500 rows/participant anyway so this
            mainly guards against very uneven participants.
--skip_bad  Drop erroneous rows for participants 2, 7, 10 (default: True)
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.io
from tqdm import tqdm

# MediaPipe — same version the tracker uses
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ──────────────────────────────────────────────────────────────────────────────
# Landmark index constants  (identical to eye_follower_v3.py)
# ──────────────────────────────────────────────────────────────────────────────
IRIS_LEFT        = [474, 475, 476, 477]
IRIS_RIGHT       = [469, 470, 471, 472]
EYE_LEFT_OUTER   = 33
EYE_LEFT_INNER   = 133
EYE_LEFT_TOP     = [159, 160, 161]
EYE_LEFT_BOTTOM  = [145, 144, 163]
EYE_RIGHT_OUTER  = 362
EYE_RIGHT_INNER  = 263
EYE_RIGHT_TOP    = [386, 387, 388]
EYE_RIGHT_BOTTOM = [374, 373, 390]

_FACE_OVAL_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454,
                  323, 361, 288, 397, 365, 379, 378, 400, 377,
                  152, 148, 176, 149, 150, 136, 172, 58,  132,
                  93,  234, 127, 162, 21,  54,  103, 67,  109]

_HEAD_IDX = [1, 152, 33, 263, 61, 291]
_HEAD_3D  = np.array([
    [  0.0,    0.0,    0.0],
    [  0.0, -330.0,  -65.0],
    [-225.0,  170.0, -135.0],
    [ 225.0,  170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [ 150.0, -150.0, -125.0],
], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Functions copied verbatim from eye_follower_v3.py
# (keeping them identical ensures feature vectors are comparable at runtime)
# ──────────────────────────────────────────────────────────────────────────────
def get_head_pose(lm_norm, img_w, img_h):
    pts2d = np.array(
        [(lm_norm[i].x * img_w, lm_norm[i].y * img_h) for i in _HEAD_IDX],
        dtype=np.float64)
    fl      = float(img_w)
    cam_mat = np.array([[fl,0,img_w/2],[0,fl,img_h/2],[0,0,1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        _HEAD_3D, pts2d, cam_mat, np.zeros((4,1)),
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        yaw   = np.degrees(np.arctan2(-R[2,0], sy))
        pitch = np.degrees(np.arctan2( R[2,1], R[2,2]))
        roll  = np.degrees(np.arctan2( R[1,0], R[0,0]))
    else:
        pitch = np.degrees(np.arctan2(-R[2,0], sy))
        yaw, roll = 0.0, np.degrees(np.arctan2(-R[0,1], R[1,1]))
    return yaw, pitch, roll, R


def get_face_scale(lm_norm):
    l, r = lm_norm[33], lm_norm[263]
    return float(np.hypot(r.x - l.x, r.y - l.y))


def get_face_centre(lm_norm):
    xs = [lm_norm[i].x for i in _FACE_OVAL_IDX]
    ys = [lm_norm[i].y for i in _FACE_OVAL_IDX]
    return (min(xs)+max(xs))*0.5, (min(ys)+max(ys))*0.5


def get_nose_tip(lm_norm):
    return lm_norm[1].x, lm_norm[1].y


def get_compensated_gaze(lm_px, R, img_w, img_h):
    nose = np.array(lm_px[1], dtype=np.float64)
    def iris_center(indices):
        return np.mean([np.array(lm_px[i], dtype=np.float64) for i in indices], axis=0)
    lc = iris_center(IRIS_LEFT)
    rc = iris_center(IRIS_RIGHT)
    face_w = np.hypot(lm_px[263][0]-lm_px[33][0],
                      lm_px[263][1]-lm_px[33][1]) + 1e-6
    def compensate(pt):
        v  = np.array([(pt[0]-nose[0])/face_w, (pt[1]-nose[1])/face_w, 0.0])
        vc = R.T @ v
        return float(vc[0]), float(vc[1])
    lx, ly = compensate(lc)
    rx, ry = compensate(rc)
    return lx, ly, rx, ry


def get_avg_iris(lm_norm):
    li = [lm_norm[i] for i in IRIS_LEFT]
    ri = [lm_norm[i] for i in IRIS_RIGHT]
    ix = sum(p.x for p in li+ri) / (len(li)+len(ri))
    iy = sum(p.y for p in li+ri) / (len(li)+len(ri))
    return ix, iy


def get_eye_ear(landmarks, outer, inner, top_ids, bottom_ids):
    lx, ly = landmarks[outer]
    rx, ry = landmarks[inner]
    eye_w  = np.hypot(rx-lx, ry-ly)
    top_y  = sum(landmarks[i][1] for i in top_ids)    / len(top_ids)
    bot_y  = sum(landmarks[i][1] for i in bottom_ids) / len(bottom_ids)
    eye_h  = abs(bot_y - top_y)
    return eye_h / (eye_w + 1e-6)


def build_feature(lx, ly, rx, ry,
                  yaw, pitch, roll, face_scale,
                  fc_x, fc_y, nose_x, nose_y, avg_ix, avg_iy):
    yn  = yaw   / 30.0
    pn  = pitch / 20.0
    rn  = roll  / 20.0
    fc_dx = fc_x - 0.5
    fc_dy = fc_y - 0.5
    return np.array([
        lx, ly, rx, ry,
        (lx+rx)*0.5, (ly+ry)*0.5,
        fc_x, fc_y, fc_dx, fc_dy,
        nose_x, nose_y,
        yn, pn, rn,
        face_scale,
        avg_ix, avg_iy,
        lx*yn,  rx*yn,
        ly*pn,  ry*pn,
        lx*rn,  rx*rn,
        fc_dx*avg_ix,
        fc_dy*avg_iy,
        fc_dx*yn,
        fc_dy*pn,
        lx*avg_ix, rx*avg_ix,
        ly*avg_iy, ry*avg_iy,
    ], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe detector  (IMAGE mode — no async needed for offline processing)
# ──────────────────────────────────────────────────────────────────────────────
def make_detector(model_path='face_landmarker.task'):
    if not Path(model_path).exists():
        print(f"[ERROR] face_landmarker.task not found at: {model_path}")
        print("Download it from:")
        print("  https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/latest/face_landmarker.task")
        sys.exit(1)
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        num_faces=1,
        min_face_detection_confidence=0.3,   # lower threshold — dataset images are cropped
        min_tracking_confidence=0.3,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


# ──────────────────────────────────────────────────────────────────────────────
# Load screen size from MATLAB .mat file
# ──────────────────────────────────────────────────────────────────────────────
def load_screen_size(calib_dir):
    """Returns (width_px, height_px) for this participant."""
    mat_path = Path(calib_dir) / 'screenSize.mat'
    if not mat_path.exists():
        # Some older copies use lowercase
        mat_path = Path(calib_dir) / 'ScreenSize.mat'
    mat = scipy.io.loadmat(str(mat_path))
    # Fields may be nested in a struct; try common layouts
    try:
        w = int(np.squeeze(mat['width_pixel']))
        h = int(np.squeeze(mat['height_pixel']))
    except KeyError:
        # Older format: fields inside a 'screenSize' struct variable
        ss = mat.get('screenSize', mat.get('ScreenSize'))
        w  = int(ss['width_pixel'][0,0])
        h  = int(ss['height_pixel'][0,0])
    return w, h


# ──────────────────────────────────────────────────────────────────────────────
# Process one participant
# ──────────────────────────────────────────────────────────────────────────────
def process_participant(pid, dataset_root, detector,
                        max_rows, skip_bad):
    p_dir  = Path(dataset_root) / f'p{pid:02d}'
    ann_file = p_dir / f'p{pid:02d}.txt'
    calib_dir = p_dir / 'Calibration'

    if not ann_file.exists():
        print(f"  [SKIP] annotation file not found: {ann_file}")
        return [], []

    # Screen size for this participant
    try:
        scr_w, scr_h = load_screen_size(calib_dir)
    except Exception as e:
        print(f"  [WARN] Could not read screenSize for p{pid:02d}: {e}")
        print(f"         Falling back to 1280×800")
        scr_w, scr_h = 1280, 800

    lines = ann_file.read_text().strip().splitlines()
    if max_rows and len(lines) > max_rows:
        rng = np.random.default_rng(seed=pid)
        lines = rng.choice(lines, size=max_rows, replace=False).tolist()

    features, targets = [], []
    skipped_oob = skipped_mp = skipped_pose = skipped_blink = 0

    for line in tqdm(lines, desc=f'  p{pid:02d}', leave=False):
        parts = line.split()
        # dim 1: relative image path (e.g. day01/00001.jpg)
        rel_path   = parts[0]
        # dim 2-3: gaze x, y in screen pixels
        gaze_x_px  = float(parts[1])
        gaze_y_px  = float(parts[2])

        # Normalise gaze to [0,1]
        gaze_x_n = gaze_x_px / scr_w
        gaze_y_n = gaze_y_px / scr_h

        # Drop out-of-screen points (known errors for p02, p07, p10)
        if skip_bad and (gaze_x_n < 0 or gaze_x_n > 1 or
                         gaze_y_n < 0 or gaze_y_n > 1):
            skipped_oob += 1
            continue

        img_path = p_dir / rel_path
        if not img_path.exists():
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result   = detector.detect(mp_image)

        if not result.face_landmarks:
            skipped_mp += 1
            continue

        lm_norm = result.face_landmarks[0]
        lm_px   = [(lm.x * w, lm.y * h) for lm in lm_norm]

        # Head pose
        pose = get_head_pose(lm_norm, w, h)
        if pose is None:
            skipped_pose += 1
            continue
        yaw, pitch, roll, R = pose

        # Blink check
        ear_l = get_eye_ear(lm_px, EYE_LEFT_OUTER, EYE_LEFT_INNER,
                            EYE_LEFT_TOP, EYE_LEFT_BOTTOM)
        ear_r = get_eye_ear(lm_px, EYE_RIGHT_OUTER, EYE_RIGHT_INNER,
                            EYE_RIGHT_TOP, EYE_RIGHT_BOTTOM)
        if ear_l < 0.08 or ear_r < 0.08:
            skipped_blink += 1
            continue

        lrx, lry, rrx, rry = get_compensated_gaze(lm_px, R, w, h)
        fs                  = get_face_scale(lm_norm)
        fc_x, fc_y          = get_face_centre(lm_norm)
        nx, ny              = get_nose_tip(lm_norm)
        aix, aiy            = get_avg_iris(lm_norm)

        feat = build_feature(
            lrx, lry, rrx, rry,
            yaw, pitch, roll, fs,
            fc_x, fc_y, nx, ny, aix, aiy)

        features.append(feat)
        targets.append([gaze_x_n, gaze_y_n])

    kept = len(features)
    total = len(lines)
    print(f"  p{pid:02d}: {kept}/{total} kept  "
          f"(oob={skipped_oob} mp_fail={skipped_mp} "
          f"pose_fail={skipped_pose} blink={skipped_blink})")
    return features, targets


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description='Preprocess MPIIFaceGaze → eye_follower_v3 features')
    ap.add_argument('--dataset', required=True,
                    help='Root folder of MPIIFaceGaze (contains p00..p14)')
    ap.add_argument('--output',  default='pretrain_features.npz',
                    help='Output .npz path (default: pretrain_features.npz)')
    ap.add_argument('--max_per_participant', type=int, default=2500,
                    help='Max rows per participant (default: 2500)')
    ap.add_argument('--skip_bad', action='store_true', default=True,
                    help='Drop out-of-screen rows for p02/p07/p10 (default: on)')
    ap.add_argument('--model', default='face_landmarker.task',
                    help='Path to face_landmarker.task (default: ./face_landmarker.task)')
    ap.add_argument('--participants', default='all',
                    help='Comma-separated participant IDs to process, e.g. 0,1,3 '
                         '(default: all)')
    args = ap.parse_args()

    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        print(f"[ERROR] Dataset path does not exist: {dataset_root}")
        sys.exit(1)

    if args.participants == 'all':
        pids = list(range(15))
    else:
        pids = [int(x) for x in args.participants.split(',')]

    print(f"MPIIFaceGaze preprocessor")
    print(f"  Dataset : {dataset_root}")
    print(f"  Output  : {args.output}")
    print(f"  Max/part: {args.max_per_participant}")
    print(f"  Subjects: {pids}")
    print(f"  Model   : {args.model}")
    print()

    detector = make_detector(args.model)

    all_features    = []
    all_targets     = []
    all_participant = []

    for pid in pids:
        feats, tgts = process_participant(
            pid, dataset_root, detector,
            args.max_per_participant, args.skip_bad)
        all_features.extend(feats)
        all_targets.extend(tgts)
        all_participant.extend([pid] * len(feats))

    if not all_features:
        print("[ERROR] No features extracted. Check dataset path and model file.")
        sys.exit(1)

    F = np.array(all_features,    dtype=np.float64)
    T = np.array(all_targets,     dtype=np.float64)
    P = np.array(all_participant, dtype=np.int32)

    print(f"\nTotal samples : {len(F)}")
    print(f"Feature dim   : {F.shape[1]}")
    print(f"Target range  : x=[{T[:,0].min():.3f},{T[:,0].max():.3f}]  "
          f"y=[{T[:,1].min():.3f},{T[:,1].max():.3f}]")

    np.savez_compressed(args.output, features=F, targets=T, participants=P)
    print(f"\nSaved → {args.output}")
    print("You can now use this file in eye_follower_v3.py with --pretrain flag.")
    print("See pretrain_and_run.py for the full fine-tuning workflow.")

    detector.close()


if __name__ == '__main__':
    main()
