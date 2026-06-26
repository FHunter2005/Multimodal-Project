import time

class DialogController:
    def __init__(self, dwell=5.0, cooldown=20.0):
        self.struggle_frames = 0
        self.help_active = False
        self.system_message = "Listening to User State..."
        
        self.dwell = dwell
        self.cooldown = cooldown
        self.cur_para = None
        self.para_t0 = None
        self.para_fired = {}
        self.para_fv = False

    def evaluate(self, mouse_data, gaze_data, epistemic_state, emotion_scores, current_para):
        # 1. Unpack Modality Data
        mouse_state = mouse_data["state"] if mouse_data else "Inactive"
        mouse_intensity = mouse_data["intensity"] if mouse_data else 0.0
        
        gaze_state = gaze_data.get("state", "Initializing")
        gaze_struggle_score = gaze_data.get("score", 0.0)
        
        confusion = epistemic_state.get('confusion', 0.0) if isinstance(epistemic_state, dict) else 0.0
        anger = emotion_scores.get('anger', 0.0) if isinstance(emotion_scores, dict) else 0.0
        sadness = emotion_scores.get('sadness', 0.0) if isinstance(emotion_scores, dict) else 0.0
        plutchik_frustration = (anger * 0.70) + (sadness * 0.30)
        face_struggle = max(confusion, plutchik_frustration)

        # 2. EVIDENCE ACCUMULATION (Late Fusion)
        # Lists hold intensity "votes" from active modalities. Empty lists won't drag down averages.
        evidence_pool = {
            "Reading (Focused)": [],
            "Skimming": [],
            "Distracted": [],
            "Struggling / Stuck": [],
            "Thinking (Off-Text)": [],
            "Deep Focus": []
        }

        # -- Cast Face Modality Votes --
        if face_struggle > 0.25:
            evidence_pool["Struggling / Stuck"].append(face_struggle)

        # -- Cast Mouse Modality Votes --
        if mouse_state in evidence_pool and mouse_state not in ["Inactive", "Undefined"]:
            evidence_pool[mouse_state].append(mouse_intensity)

        # -- Cast Gaze Modality Votes --
        if gaze_state in evidence_pool:
            if gaze_state == "Struggling / Stuck":
                evidence_pool[gaze_state].append(gaze_struggle_score)
            elif gaze_state == "Reading (Focused)":
                evidence_pool[gaze_state].append(1.0 - gaze_struggle_score) # Inverse of struggle
            else:
                evidence_pool[gaze_state].append(0.8) # Baseline strong confidence for other gaze states

        # 3. CALCULATE FUSED STATE
        fused_state = "Reading (Focused)" # Fallback default
        fused_intensity = 0.0
        
        best_avg = -1.0
        for state, votes in evidence_pool.items():
            if len(votes) > 0:
                avg_intensity = sum(votes) / len(votes)
                if avg_intensity > best_avg:
                    best_avg = avg_intensity
                    fused_state = state
                    fused_intensity = avg_intensity

        # Override Base Struggle if purely in Deep Focus
        if fused_state == "Deep Focus" and face_struggle < 0.5:
            fused_state = "Deep Focus"

        # 4. TEMPORAL THRESHOLDING
        if fused_state == "Struggling / Stuck" and fused_intensity > 0.40:
            self.struggle_frames += 1
        else:
            self.struggle_frames = max(0, self.struggle_frames - 2)

        TRIGGER_THRESHOLD = 300 
        triggered_para = None
        needs_summary = False
        now = time.time()
        
        # 5. PARAGRAPH DWELL LOGIC
        if current_para != self.cur_para:
            self.cur_para = current_para
            self.para_t0 = now if current_para is not None else None
            self.para_fv = False
        elif current_para is not None and not self.para_fv:
            
            dynamic_dwell = self.dwell * 0.5 if (fused_state == "Struggling / Stuck") else self.dwell
            
            if now - self.para_t0 >= dynamic_dwell and now - self.para_fired.get(current_para, 0) >= self.cooldown:
                self.para_fired[current_para] = now
                self.para_fv = True
                
                # --- DECISION ENGINE ---
                if plutchik_frustration > 0.6 and plutchik_frustration > confusion:
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
        
        # 6. GLOBAL STATE INTERVENTIONS
        if self.struggle_frames > TRIGGER_THRESHOLD and not self.help_active:
            self.help_active = True
            if plutchik_frustration > confusion: self.system_message = "INTERVENTION: High Frustration. Take a short break."
            elif fused_state == "Thinking (Off-Text)": self.system_message = "INTERVENTION: Processing information. No rush."
            elif fused_intensity > 0.6: self.system_message = "INTERVENTION: High Cognitive Load detected. Relax your eyes."
            else: self.system_message = "INTERVENTION: Confusion detected. Reading pace adjusted."
                
        elif self.struggle_frames == 0 and self.help_active and not needs_summary:
            self.help_active = False
            self.system_message = f"User engaged. Normal reading. ({fused_state})"

        return {
            "help_active": self.help_active,
            "message": self.system_message,
            "struggle_level": min(1.0, self.struggle_frames / float(TRIGGER_THRESHOLD)),
            "fused_state": fused_state,
            "fused_intensity": fused_intensity,
            "triggered_para": triggered_para,
            "needs_summary": needs_summary
        }

    def reset_intervention(self):
        self.struggle_frames = 0
        self.help_active = False
        self.reset_dwell()
        
    def reset_dwell(self):
        self.cur_para = None
        self.para_t0 = None
        self.para_fv = False