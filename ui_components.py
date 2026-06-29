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

class GazeHeatmap:
    def __init__(self, width=SCREEN_W, height=SCREEN_H, blob_sigma=55.0, decay=0.8, alpha=0.55):
        self.w, self.h, self.decay, self.alpha, self.acc = width, height, decay, alpha, np.zeros((height, width), dtype=np.float32)
        r = int(blob_sigma * 3.5); ax = np.arange(-r, r + 1, dtype=np.float32); xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * blob_sigma**2)); self._kernel, self._r = (kernel / kernel.max()).astype(np.float32), r
        
    def update(self, gx: int, gy: int):
        self.acc *= self.decay
        x0, x1, y0, y1 = max(gx - self._r, 0), min(gx + self._r + 1, self.w), max(gy - self._r, 0), min(gy + self._r + 1, self.h)
        kx0, ky0 = x0 - (gx - self._r), y0 - (gy - self._r)
        self.acc[y0:y1, x0:x1] += self._kernel[ky0:ky0 + (y1 - y0), kx0:kx0 + (x1 - x0)]
        
    def render(self, canvas: np.ndarray) -> np.ndarray:
        peak = self.acc.max()
        if peak < 1e-3: return canvas
        norm = np.clip(self.acc / peak, 0.0, 1.0)
        alpha_map = (norm * self.alpha * (norm > 0.05).astype(np.float32))
        color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        for c in range(3): canvas[:, :, c] = np.clip(color[:, :, c] * alpha_map + canvas[:, :, c] * (1.0 - alpha_map), 0, 255).astype(np.uint8)
        return canvas
        
    def reset(self): self.acc[:] = 0.0

