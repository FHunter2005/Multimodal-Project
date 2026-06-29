import time

class DialogController:
    def __init__(self, dwell=5.0, cooldown=20.0, calibration_duration=30.0, tip_cooldown=30.0):
        self.struggle_frames = 0
        self.help_active = False
        self.system_message = "Listening to User State..."
        
        self.dwell = dwell
        self.cooldown = cooldown
        self.cur_para = None
        self.para_t0 = None
        self.para_fired = {}
        self.para_fv = False
        
        self.recent_mouse_frames = 0
        self.start_time = None
        self.calibration_duration = calibration_duration
        self.is_calibrating = True
        
        self.tip_cooldown = tip_cooldown
        self.last_tip_time = 0.0
        
        # Data buffers to establish baseline (Mouse only now)
        self.calib_mouse_intensities = []
        
        # Dynamic User Profile (Facial baselines removed)
        self.user_profile = {
            "is_mouse_reader": True
        }
        
        self.state_alpha = 0.05  # Tuning parameter (0.01 to 0.1). Lower = slower state transitions.
        self.current_fused_state = "Reading (Focused)"
        self.smoothed_intensities = {
            "Reading (Focused)": 0.0,
            "Skimming": 0.0,
            "Distracted": 0.0,
            "Struggling / Stuck": 0.0,
            "Thinking (Off-Text)": 0.0,
            "Deep Focus": 0.0
        }

    def _finalize_calibration(self):
        """Calculates baselines and creates the user profile."""
        self.is_calibrating = False
        
        valid_mouse_moves = len([m for m in self.calib_mouse_intensities if m > 0.0])
        if valid_mouse_moves < (self.calibration_duration / 2):
            self.user_profile["is_mouse_reader"] = False
            
        print(f"[SYSTEM] Calibration Complete. Profile generated: {self.user_profile}")
        self.system_message = "Calibration complete. Monitoring active."

    def _update_dynamic_profile(self, mouse_intensity):
        """Continuously adapts the user profile to behavioral drift."""
        if not self.user_profile["is_mouse_reader"]:
            if mouse_intensity > 0.1:
                self.recent_mouse_frames += 1
            else:
                self.recent_mouse_frames = max(0, self.recent_mouse_frames - 2)
                
            if self.recent_mouse_frames > 50: 
                self.user_profile["is_mouse_reader"] = True
                print("[SYSTEM] User behavior shifted: Mouse Tracking ACTIVATED.")

    def evaluate(self, mouse_data, gaze_data, epistemic_state, emotion_scores, current_para):
        now = time.time()
        
        if self.start_time is None:
            self.start_time = now

        in_cooldown = (now - self.last_tip_time) < self.tip_cooldown

        # 1. Unpack Modality Data
        mouse_state = mouse_data["state"] if mouse_data else "Inactive"
        mouse_intensity = mouse_data["intensity"] if mouse_data else 0.0
        
        gaze_state = gaze_data.get("state", "Initializing")
        gaze_struggle_score = gaze_data.get("score", 0.0)
        gaze_y = gaze_data.get("y", 540)

        confusion = epistemic_state.get('confusion', 0.0) if isinstance(epistemic_state, dict) else 0.0
        anger = emotion_scores.get('anger', 0.0) if isinstance(emotion_scores, dict) else 0.0
        sadness = emotion_scores.get('sadness', 0.0) if isinstance(emotion_scores, dict) else 0.0
        
        plutchik_frustration = (anger * 0.70) + (sadness * 0.30)
        face_struggle = max(confusion, plutchik_frustration)

        # ---------------------------------------------------------
        # 2. CALIBRATION PHASE (Observation Only)
        # ---------------------------------------------------------
        if self.is_calibrating:
            if now - self.start_time < self.calibration_duration:
                self.calib_mouse_intensities.append(mouse_intensity)
                
                return {
                    "help_active": False,
                    "message": "CALIBRATING: Establishing your baseline reading style...",
                    "struggle_level": 0.0,
                    "fused_state": "Calibrating",
                    "fused_intensity": 0.0,
                    "triggered_para": None,
                    "needs_summary": False,
                    "face_score": face_struggle,
                    "profile": self.user_profile
                }
            else:
                self._finalize_calibration()

        self._update_dynamic_profile(mouse_intensity)

        # ---------------------------------------------------------
        # 3. EVIDENCE ACCUMULATION (Profile-Aware Fusion)
        # ---------------------------------------------------------
        evidence_pool = {
            "Reading (Focused)": [],
            "Skimming": [],
            "Distracted": [],
            "Struggling / Stuck": [],
            "Thinking (Off-Text)": [],
            "Deep Focus": []
        }

        # Directly use raw face_struggle score instead of adjusted baselines
        if face_struggle > 0.25:
            evidence_pool["Struggling / Stuck"].append(min(1.0, face_struggle))

        if self.user_profile["is_mouse_reader"]:
            if mouse_state in evidence_pool and mouse_state not in ["Inactive", "Undefined"]:
                evidence_pool[mouse_state].append(mouse_intensity)

        if gaze_state in evidence_pool:
            if gaze_state == "Struggling / Stuck":
                evidence_pool[gaze_state].append(gaze_struggle_score)
            elif gaze_state == "Reading (Focused)":
                bottom_penalty = min(0.3, (gaze_y - 750) / 1000.0) if gaze_y > 750 else 0.0
                intensity = max(0.1, (1.0 - gaze_struggle_score) - bottom_penalty)
                evidence_pool[gaze_state].append(intensity) 
            elif gaze_state == "Deep Focus":
                bottom_penalty = min(0.4, (gaze_y - 750) / 1000.0) if gaze_y > 750 else 0.0
                evidence_pool[gaze_state].append(max(0.1, 0.8 - bottom_penalty))
            else:
                evidence_pool[gaze_state].append(0.8)

        # ---------------------------------------------------------
        # 4. CALCULATE FUSED STATE (WITH TEMPORAL SMOOTHING)
        # ---------------------------------------------------------
        instant_intensities = {k: 0.0 for k in evidence_pool.keys()}
        for state, votes in evidence_pool.items():
            if len(votes) > 0:
                instant_intensities[state] = sum(votes) / len(votes)

        best_smoothed_val = -1.0
        new_fused_state = self.current_fused_state
        new_fused_intensity = 0.0
        
        for state in self.smoothed_intensities.keys():
            self.smoothed_intensities[state] = (self.state_alpha * instant_intensities[state]) + ((1.0 - self.state_alpha) * self.smoothed_intensities[state])
            
            if self.smoothed_intensities[state] > best_smoothed_val:
                best_smoothed_val = self.smoothed_intensities[state]
                new_fused_state = state
                new_fused_intensity = self.smoothed_intensities[state]

        # Handle specific overrides using raw face_struggle
        if new_fused_state == "Deep Focus" and face_struggle < 0.5:
            pass # Keep it as Deep Focus

        self.current_fused_state = new_fused_state
        fused_state = new_fused_state
        fused_intensity = new_fused_intensity

        # ---------------------------------------------------------
        # 5. TEMPORAL THRESHOLDING
        # ---------------------------------------------------------
        if fused_state == "Struggling / Stuck" and fused_intensity > 0.40:
            self.struggle_frames += 1
        else:
            self.struggle_frames = max(0, self.struggle_frames - 2)

        TRIGGER_THRESHOLD = 300 
        triggered_para = None
        needs_summary = False
        
        # ---------------------------------------------------------
        # 6. PARAGRAPH DWELL LOGIC
        # ---------------------------------------------------------
        if current_para != self.cur_para:
            self.cur_para = current_para
            self.para_t0 = now if current_para is not None else None
            self.para_fv = False
        elif current_para is not None and not self.para_fv:
            
            dynamic_dwell = self.dwell * 0.5 if (fused_state == "Struggling / Stuck") else self.dwell
            
            if now - self.para_t0 >= dynamic_dwell and now - self.para_fired.get(current_para, 0) >= self.cooldown:
                self.para_fired[current_para] = now
                self.para_fv = True
                
                # --- DECISION ENGINE (GATED BY COOLDOWN) ---
                if not in_cooldown:
                    self.last_tip_time = now 
                    # Using raw face_struggle here
                    if plutchik_frustration > 0.6 and face_struggle > 0.3:
                        self.help_active = True
                        self.system_message = "INTERVENTION: High Frustration. Take a short break, don't force it."
                    elif fused_state == "Thinking (Off-Text)":
                        self.help_active = True
                        self.system_message = "INTERVENTION: Taking time to process. Let me know if you need help."
                    else:
                        triggered_para = current_para
                        needs_summary = True
                        self.help_active = True
                        self.system_message = "INTERVENTION: Cognitive load detected. Generating paragraph summary..."
        
        # ---------------------------------------------------------
        # 7. GLOBAL STATE INTERVENTIONS
        # ---------------------------------------------------------
        if self.struggle_frames > TRIGGER_THRESHOLD and not self.help_active:
            if not in_cooldown:
                self.last_tip_time = now 
                self.help_active = True
                if plutchik_frustration > confusion: 
                    self.system_message = "INTERVENTION: High Frustration. Take a short break."
                elif fused_state == "Thinking (Off-Text)": 
                    self.system_message = "INTERVENTION: Processing information. No rush."
                elif fused_intensity > 0.6: 
                    self.system_message = "INTERVENTION: High Cognitive Load detected. Relax your eyes."
                else: 
                    self.system_message = "INTERVENTION: Confusion detected. Reading pace adjusted."
                
        elif self.struggle_frames == 0 and self.help_active and not needs_summary:
            self.help_active = False
            self.system_message = f"User engaged. Normal reading. ({fused_state.lower()})"
            self.last_tip_time = now 
        
        return {
            "help_active": self.help_active,
            "message": self.system_message,
            "struggle_level": min(1.0, self.struggle_frames / float(TRIGGER_THRESHOLD)),
            "fused_state": fused_state,
            "fused_intensity": fused_intensity,
            "triggered_para": triggered_para,
            "needs_summary": needs_summary,
            "face_score": face_struggle,
            "profile": self.user_profile
        }

    def reset_intervention(self):
        self.struggle_frames = 0
        self.help_active = False
        self.last_tip_time = time.time()
        self.reset_dwell()
        
    def reset_dwell(self):
        self.cur_para = None
        self.para_t0 = None
        self.para_fv = False