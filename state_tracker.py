import cv2
import tempfile
import base64
import requests
import threading
import os
import numpy as np

class HuggingFaceDAiSEETracker:
    """
    Collects live webcam frames, chunks them into short video clips, 
    and sends them to a Hugging Face Inference Endpoint for engagement detection.
    """
    def __init__(self, endpoint_url, hf_token, frames_per_chunk=30, fps=15):
        self.endpoint_url = endpoint_url
        self.hf_token = hf_token
        self.frames_per_chunk = frames_per_chunk
        self.fps = fps
        
        self.frame_buffer = []
        self.lock = threading.Lock()
        self.is_processing = False
        
        # DAiSEE standard labels
        self.emotions = ["Engagement", "Confusion", "Frustration", "Boredom"]
        self.target_scores = {e: 0.0 for e in self.emotions}
        self.smooth_scores = {e: 0.0 for e in self.emotions}
        self.alpha = 0.15 # UI animation smoothing

        self.colors = {
            "Engagement":  (100, 255, 100), # Green
            "Confusion":   (50, 150, 255),  # Orange
            "Frustration": (50, 50, 220),   # Red
            "Boredom":     (150, 150, 150)  # Gray
        }

    def update_frame(self, frame):
        """Called every frame in your main while loop."""
        if not self.endpoint_url or not self.hf_token:
            return # Skip if API keys aren't set

        with self.lock:
            # Resize frame heavily to save bandwidth and processing time
            # 224x224 is the standard input size for most video models (like VideoMAE)
            small_frame = cv2.resize(frame, (224, 224))
            self.frame_buffer.append(small_frame)
            
            # When we have enough frames for a chunk, spawn a thread to send it
            if len(self.frame_buffer) >= self.frames_per_chunk and not self.is_processing:
                chunk = list(self.frame_buffer)
                self.frame_buffer.clear()
                self.is_processing = True
                threading.Thread(target=self._process_chunk, args=(chunk,), daemon=True).start()

    def _process_chunk(self, frames):
        """Background thread: converts frames to MP4, Base64 encodes, and calls API."""
        temp_filename = None
        try:
            # 1. Create a temporary video file on disk
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                temp_filename = tmp.name
            
            # 2. Write the frames to the temp mp4 file using OpenCV
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_filename, fourcc, self.fps, (224, 224))
            for f in frames:
                out.write(f)
            out.release()
            
            # 3. Base64 Encode the temp file
            with open(temp_filename, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode()
            
            # 4. Call Hugging Face API
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            payload = {"inputs": video_b64}
            response = requests.post(self.endpoint_url, headers=headers, json=payload, timeout=8.0)
            
            # 5. Parse Response
            if response.status_code == 200:
                predictions = response.json()
                # HF models usually return a list: [{"label": "Confusion", "score": 0.82}, ...]
                if isinstance(predictions, list):
                    for item in predictions:
                        label = item.get("label", "")
                        if label in self.target_scores:
                            self.target_scores[label] = item.get("score", 0.0)
            else:
                print(f"[HF API Error] {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"[Chunk Processing Error] {e}")
        finally:
            # Clean up the temp file and free the lock
            if temp_filename and os.path.exists(temp_filename):
                os.remove(temp_filename)
            self.is_processing = False

    def render(self, canvas_640x540):
        """Draws the live updating bar chart in the bottom right corner."""
        h, w = canvas_640x540.shape[:2]
        
        cv2.rectangle(canvas_640x540, (0, 0), (w, h), (20, 20, 20), -1)
        cv2.putText(canvas_640x540, "Epistemic State (Hugging Face DAiSEE)", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
        
        start_y, bar_height, gap, max_bar_width = 100, 40, 40, w - 200

        for i, emotion in enumerate(self.emotions):
            # Smooth the scores so the UI bars don't jitter
            self.smooth_scores[emotion] = (self.smooth_scores[emotion] * (1 - self.alpha) + 
                                           self.target_scores[emotion] * self.alpha)
            score = self.smooth_scores[emotion]
            color = self.colors[emotion]
            
            y = start_y + i * (bar_height + gap)
            
            cv2.putText(canvas_640x540, emotion, (20, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.rectangle(canvas_640x540, (160, y), (160 + max_bar_width, y + bar_height), (40, 40, 40), -1)
            cv2.rectangle(canvas_640x540, (160, y), (160 + int(max_bar_width * score), y + bar_height), color, -1)
            cv2.putText(canvas_640x540, f"{score:.2f}", (160 + int(max_bar_width * score) + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return canvas_640x540