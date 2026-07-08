import speech_recognition as sr
import threading

class VoiceListener:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.is_listening = False
        self.result = None
        
    def listen_async(self):
        """Starts listening in a background thread immediately."""
        if self.is_listening:
            return
        self.is_listening = True
        self.result = None
        threading.Thread(target=self._listen_worker, daemon=True).start()

    def _listen_worker(self):
        try:
            with sr.Microphone() as source:
                print("\n[Voice] System finished speaking. Mic active. Listening for Yes/No...")
                # Quickly adjust for noise, then listen
                self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                audio = self.recognizer.listen(source, timeout=4.0, phrase_time_limit=3.0)
            
            print("[Voice] Processing speech...")
            text = self.recognizer.recognize_google(audio).lower()
            print(f"[Voice] Heard: '{text}'")
            
            if any(w in text for w in ["yes", "yeah", "sure", "yep", "ok", "please"]):
                self.result = 'Y'
            elif any(w in text for w in ["no", "nope", "nah", "stop", "cancel"]):
                self.result = 'N'
            else:
                self.result = 'UNKNOWN'
                
        except sr.WaitTimeoutError:
            print("[Voice] Listening timed out. No speech detected.")
            self.result = 'TIMEOUT'
        except Exception as e:
            print(f"[Voice] Error: {e}")
            self.result = 'ERROR'
        finally:
            self.is_listening = False

    def get_result(self):
        res = self.result
        if res is not None:
            self.result = None 
        return res