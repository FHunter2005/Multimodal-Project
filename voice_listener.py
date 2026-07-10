import speech_recognition as sr

class VoiceListener:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
        # 1. Print available mics so you can verify it's using the right one
        print("\n--- Audio Debug ---")
        mics = sr.Microphone.list_microphone_names()
        for i, mic_name in enumerate(mics):
            print(f"Mic {i}: {mic_name}")
        print("-------------------\n")
        
        # If the default mic is wrong, you can force an index like this:
        # self.mic = sr.Microphone(device_index=1) 
        self.mic = sr.Microphone() 
        
        self.stop_listening_fn = None
        self.app_callback = None
        
    def start_continuous(self, callback):
        """Starts a background thread that constantly listens for speech."""
        self.app_callback = callback
        with self.mic as source:
            print("[Voice] Calibrating to ambient noise (1 sec)...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
            
            # 2. Prevent the threshold from being set too high
            print(f"[Voice] Auto-calibrated threshold: {self.recognizer.energy_threshold}")
            self.recognizer.energy_threshold = min(self.recognizer.energy_threshold, 400)
            self.recognizer.dynamic_energy_threshold = False # Stop it from drifting
            print(f"[Voice] Final threshold locked at: {self.recognizer.energy_threshold}")
            
        print("[Voice] Continuous listening activated. Say 'summarize this'...")
        
        self.stop_listening_fn = self.recognizer.listen_in_background(
            self.mic, 
            self._audio_callback,
            phrase_time_limit=4.0
        )

    def _audio_callback(self, recognizer, audio):
        """Triggered automatically by the background thread when audio is captured."""
        try:
            # 3. Prove that the microphone actually triggered a recording
            print("[Voice] Audio captured, sending to Google...")
            
            text = recognizer.recognize_google(audio).lower()
            print(f"[Voice] Heard: '{text}'")
            
            if self.app_callback:
                self.app_callback(text)
                
        except sr.UnknownValueError:
            # 4. UN-SILENCE THE ERROR
            print("[Voice] Google heard something, but couldn't understand the words (Try speaking louder).")
        except Exception as e:
            print(f"[Voice] API Error: {e}")

    def stop(self):
        if self.stop_listening_fn:
            self.stop_listening_fn(wait_for_stop=False)