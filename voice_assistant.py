import shutil
import subprocess
import threading

class VoiceAssistant:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self._engine = None
        self._fallback_command = None
        self._initialize()

    def _initialize(self):
        if not self.enabled: return
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            return
        except Exception:
            self._engine = None

        for candidate in ("espeak", "spd-say"):
            if shutil.which(candidate):
                self._fallback_command = candidate
                return

    def speak(self, text, on_finish=None):
        """Speaks the text, then calls on_finish() when the audio is completely done."""
        if not self.enabled or not text:
            if on_finish: on_finish()
            return

        if self._engine is not None:
            threading.Thread(target=self._speak_with_engine, args=(text, on_finish), daemon=True).start()
        elif self._fallback_command is not None:
            threading.Thread(target=self._speak_with_fallback, args=(text, on_finish), daemon=True).start()
        else:
            if on_finish: on_finish()

    def _speak_with_engine(self, text, on_finish):
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception:
            pass
        finally:
            if on_finish: 
                on_finish()

    def _speak_with_fallback(self, text, on_finish):
        try:
            # Using subprocess.run blocks the thread until the speech command finishes
            subprocess.run([self._fallback_command, text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        finally:
            if on_finish: 
                on_finish()