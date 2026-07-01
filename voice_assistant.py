import shutil
import subprocess
import threading


class VoiceAssistant:
    """Small wrapper around text-to-speech so the dialog controller can prompt the user."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._engine = None
        self._fallback_command = None
        self._initialize()

    def _initialize(self):
        if not self.enabled:
            return

        try:
            import pyttsx3
        except Exception:
            pyttsx3 = None

        if pyttsx3 is not None:
            try:
                self._engine = pyttsx3.init()
                return
            except Exception:
                self._engine = None

        for candidate in ("espeak", "spd-say"):
            if shutil.which(candidate):
                self._fallback_command = candidate
                return

    def speak(self, text):
        if not self.enabled or not text:
            return

        if self._engine is not None:
            thread = threading.Thread(target=self._speak_with_engine, args=(text,), daemon=True)
            thread.start()
            return

        if self._fallback_command is not None:
            thread = threading.Thread(target=self._speak_with_fallback, args=(text,), daemon=True)
            thread.start()

    def _speak_with_engine(self, text):
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception:
            pass

    def _speak_with_fallback(self, text):
        try:
            if self._fallback_command == "espeak":
                subprocess.Popen([self._fallback_command, text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.Popen([self._fallback_command, text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
