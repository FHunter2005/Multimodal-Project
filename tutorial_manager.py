import cv2
import numpy as np

class TutorialManager:
    def __init__(self, width, height):
        self.is_active = True
        self.current_page = 0
        self.width = width
        self.height = height
        
        # Define the tutorial pages
        self.pages = [
            "Welcome to the Adaptive Reader.\n\nThis system doesn't just display text;\nit actively adapts to ensure you understand\nthe material.",
            "How it works:\nThe system uses your webcam and mouse to\npassively track eye movements, expressions,\nand cursor speed to measure cognitive load.",
            "How to use it:\nLook at a paragraph and say 'Explain this'.\nIf the system notices you struggling,\nit will proactively offer help.\n\nSay 'Next' or press [SPACE] to begin."
        ]

    def process_input(self, key_press, voice_command):
        """Advances the tutorial based on UI or Voice input."""
        if not self.is_active:
            return

        # Check for Voice Command ("next") OR Keyboard ([Space] is 32)
        voice_triggered = voice_command and "next" in voice_command.lower()
        key_triggered = key_press == 32 

        if voice_triggered or key_triggered:
            self.current_page += 1
            
            # If we pass the last page, deactivate the tutorial
            if self.current_page >= len(self.pages):
                self.is_active = False

    def _draw_text_with_newlines(self, frame, text, x, y, font, scale, color, thickness):
        """Helper to draw text with line breaks in OpenCV."""
        y0, dy = y, int(35 * scale) # Line height spacing
        for i, line in enumerate(text.split('\n')):
            y_pos = y0 + i * dy
            cv2.putText(frame, line, (x, y_pos), font, scale, color, thickness, cv2.LINE_AA)

    def draw_overlay(self, frame):
        """Draws the dimmed background and tutorial text over the main UI."""
        if not self.is_active:
            return frame

        # 1. Create a dark semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (15, 15, 15), -1)
        
        # Blend the overlay with the original frame (85% dark, 15% original UI)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

        # 2. Draw the tutorial dialog box
        box_width, box_height = 600, 300
        x_offset = (self.width - box_width) // 2
        y_offset = (self.height - box_height) // 2
        
        # Draw box background and border
        cv2.rectangle(frame, (x_offset, y_offset), (x_offset + box_width, y_offset + box_height), (40, 40, 40), -1)
        cv2.rectangle(frame, (x_offset, y_offset), (x_offset + box_width, y_offset + box_height), (200, 100, 50), 2)

        # 3. Render the current page's text
        text = self.pages[self.current_page]
        self._draw_text_with_newlines(
            frame=frame, 
            text=text, 
            x=x_offset + 30, 
            y=y_offset + 60, 
            font=cv2.FONT_HERSHEY_SIMPLEX, 
            scale=0.7, 
            color=(255, 255, 255), 
            thickness=1
        )
        
        # 4. Draw pagination indicator (e.g., "Page 1/3")
        page_indicator = f"Page {self.current_page + 1}/{len(self.pages)}"
        cv2.putText(frame, page_indicator, (x_offset + 30, y_offset + box_height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        return frame