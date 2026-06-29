"""
Unified Multimodal PDF Reader & Dashboard
====================================================================
Architecture (HYBRID FUSION TRACKING + PDF READING):
- Analyzers: Gaze (MediaPipe/ResNet), Emotion (MediaPipe), Epistemic (MediaPipe), Mouse
- Gaze features: RAW output (No Kalman, no temporal smoothing, no deadzones)
- PDF features: PyMuPDF rendering, FUSED Multimodal Stuck Detection, Gemini Summarization
"""

import cv2
import time
import numpy as np
import mediapipe as mp
import threading
import argparse
import os
import sys
from collections import deque
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import fitz
except ImportError:
    print("[ERROR] pip install pymupdf")
    sys.exit(1)

# Assume these are available in your local environment
from local_epistemic_tracker import LocalEpistemicTracker
from emotion_wheel import EmotionDetector, PlutchikWheel
from mouse_analyzer import MouseReadingAnalyzer
from gaze_analyzer import GazeReadingAnalyzer
from gaze_core import GazeReader, CALIB_PTS, DRIFT_PTS

# =============================================================
# 1. CORE CONSTANTS & UI HELPERS
# =============================================================
SCREEN_W, SCREEN_H = 1920, 1080
SANDBOX_W, SANDBOX_H, WHEEL_W, WHEEL_H = 1280, 1080, 640, 540



SAMPLES_NEEDED = 10 
BLINK_BLENDSHAPE_THRESHOLD = 0.35 
SCROLL_STEP = 120

def draw_target_dot(canvas, tx, ty, progress, label):
    cv2.ellipse(canvas, (tx, ty), (22, 22), -90, 0, int(360 * progress), (0, 255, 100), 3)
    cv2.circle(canvas, (tx, ty), 12, (0, 200, 80), -1)
    cv2.putText(canvas, label, ((SCREEN_W - 440) // 2, SCREEN_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)


# =============================================================
# 2. PDF & SUMMARIZATION MODULES
# =============================================================
# =============================================================
# 2. PDF & SUMMARIZATION MODULES
# =============================================================
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
        ph   = page.rect.height
        paras = []
        
        for b in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
            if b["type"] != 0: continue
            
            lines = b.get("lines", [])
            if not lines: continue
            
            # Ignore headers / large titles
            if len(lines) <= 2:
                avg = np.mean([s.get("size",10) for l in lines for s in l.get("spans",[])] or [10])
                if avg > 13: continue
                
            current_para = []
            cx0, cy0, cx1, cy1 = float('inf'), float('inf'), -float('inf'), -float('inf')
            prev_bbox = None
            
            for line in lines:
                lx0, ly0, lx1, ly1 = line["bbox"]
                
                # Extract text for the current line
                text = "".join(s["text"] for s in line.get("spans", [])).strip()
                if not text: continue
                
                # Check if this line should start a new paragraph
                is_new_para = False
                if prev_bbox is not None:
                    prev_lx0, _, _, prev_ly1 = prev_bbox
                    line_height = ly1 - ly0
                    gap = ly0 - prev_ly1
                    
                    # 1. Vertical Gap: Is there spacing between this line and the last?
                    if gap > (line_height * 0.4):
                        is_new_para = True
                    # 2. Indentation: Is this line indented significantly?
                    elif (lx0 - prev_lx0) > (line_height * 1.5):
                        is_new_para = True
                
                # If a new paragraph is detected, finalize the previous one and reset
                if is_new_para and current_para:
                    para_text = " ".join(current_para).strip()
                    if len(para_text) >= 80 and cy0 >= ph * 0.12:
                        paras.append({
                            "bbox": (int(cx0 * self.zoom), int(cy0 * self.zoom), 
                                     int(cx1 * self.zoom), int(cy1 * self.zoom)),
                            "text": para_text
                        })
                    current_para = []
                    cx0, cy0, cx1, cy1 = float('inf'), float('inf'), -float('inf'), -float('inf')
                
                # Accumulate the line into the current paragraph hitbox
                current_para.append(text)
                cx0, cy0 = min(cx0, lx0), min(cy0, ly0)
                cx1, cy1 = max(cx1, lx1), max(cy1, ly1)
                prev_bbox = (lx0, ly0, lx1, ly1)
            
            # Save the final accumulated paragraph from the block
            if current_para:
                para_text = " ".join(current_para).strip()
                if len(para_text) >= 80 and cy0 >= ph * 0.12:
                    paras.append({
                        "bbox": (int(cx0 * self.zoom), int(cy0 * self.zoom), 
                                 int(cx1 * self.zoom), int(cy1 * self.zoom)),
                        "text": para_text
                    })
                    
        self._cache[num] = (img, paras)
        return img, paras

    def close(self): self.doc.close()

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

def wrap_text(text, n=65):
    words = text.split(); lines = []; line = ""
    for w in words:
        if len(line)+len(w)+1 > n:
            if line: lines.append(line)
            line = w
        else: line = (line+" "+w).strip()
    if line: lines.append(line)
    return lines

def draw_dialog(canvas, dlg_summary):
    dim = canvas.copy(); canvas[:] = canvas // 3
    cv2.addWeighted(dim, 0.15, canvas, 0.85, 0, canvas)
    dw, dh = 900, 500; dx0=(SCREEN_W-dw)//2; dy0=(SCREEN_H-dh)//2
    cv2.rectangle(canvas, (dx0,dy0),    (dx0+dw,dy0+dh), (25,25,35),    -1)
    cv2.rectangle(canvas, (dx0,dy0),    (dx0+dw,dy0+dh), (100,180,255),  2)
    cv2.rectangle(canvas, (dx0,dy0),    (dx0+dw,dy0+52), (40,40,60),    -1)
    cv2.putText(canvas, "Looks like you might be stuck on this paragraph.",
                (dx0+20, dy0+36), cv2.FONT_HERSHEY_DUPLEX, 0.85, (100,210,255), 1, cv2.LINE_AA)
    cv2.line(canvas, (dx0, dy0+52), (dx0+dw, dy0+52), (100,180,255), 1)
    sumtext = dlg_summary['text']
    if sumtext is None:
        cv2.putText(canvas, "Generating AI summary...", (dx0+24, dy0+110),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (160,160,160), 1, cv2.LINE_AA)
    else:
        for i, line in enumerate(wrap_text(sumtext)[:10]):
            cv2.putText(canvas, line, (dx0+24, dy0+96+i*36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.72, (230,230,230), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (dx0+24,   dy0+dh-64), (dx0+220,  dy0+dh-18), (0,160,70),   -1)
    cv2.putText(canvas, "Y   Keep summary",       (dx0+34,   dy0+dh-34), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (dx0+dw-210,dy0+dh-64), (dx0+dw-18,dy0+dh-18), (170,50,50), -1)
    cv2.putText(canvas, "N   Dismiss",            (dx0+dw-200,dy0+dh-34), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

