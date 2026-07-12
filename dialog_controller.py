import time
from collections import deque

class DialogController:
    def __init__(self, dwell=5.0, cooldown=20.0, calibration_duration=5.0, tip_cooldown=30.0, voice_assistant=None):
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
        self.voice_assistant = voice_assistant
        self.active_reason = None

        self.off_text_frames = 0
        self.distracted_frames = 0
        self.last_voice_prompt_time = 0.0
        self.voice_prompt_cooldown = 25.0
        self.distraction_confidence = 0.0
        self.same_block_fixation_frames = 0
        self.last_fixation_block = None
        self.last_gaze_xy = None
        
        # Data buffers to establish baseline (Mouse only now)
        self.calib_mouse_intensities = []
        self.intervention_buffer = deque(maxlen=30)  # Holds ~1 second of frames at 30fps
        self.config = {
            "frust_thresh": 0.60,
            "face_struggle_thresh": 0.30,
            "consensus_required": 22  # Require 22 out of 30 frames to agree
        }
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

    def evaluate(self, mouse_data, gaze_data, epistemic_state, emotion_scores, current_para, physical_cues=None):
        now = time.time()
        
        if self.start_time is None:
            self.start_time = now

        in_cooldown = (now - self.last_tip_time) < self.tip_cooldown

        # 1. Unpack Modality Data
        mouse_state = mouse_data["state"] if mouse_data else "Inactive"
        mouse_intensity = mouse_data["intensity"] if mouse_data else 0.0
        
        gaze_state = gaze_data.get("state", "Initializing")
        gaze_struggle_score = gaze_data.get("score", 0.0)
        gaze_x = gaze_data.get("x")
        gaze_y = gaze_data.get("y", 540)

        confusion = epistemic_state.get('confusion', 0.0) if isinstance(epistemic_state, dict) else 0.0
        anger = emotion_scores.get('anger', 0.0) if isinstance(emotion_scores, dict) else 0.0
        sadness = emotion_scores.get('sadness', 0.0) if isinstance(emotion_scores, dict) else 0.0
        frustration = epistemic_state.get('frustration', 0.0) if isinstance(epistemic_state, dict) else 0.0
        epistemic_struggle = max(confusion, frustration)
        
        plutchik_frustration = (anger * 0.70) + (sadness * 0.30)
        face_struggle = max(confusion, plutchik_frustration)
        physical_cues = physical_cues or {}
        leaning_in = bool(physical_cues.get('leaning_in', False))
        squinting = bool(physical_cues.get('squinting', False))
        head_turned_away = bool(physical_cues.get('head_turned_away', False))
        face_looking_at_screen = bool(physical_cues.get('face_looking_at_screen', True))
        gaze_off_screen = bool(physical_cues.get('gaze_off_screen', False))
        gaze_wandering = bool(physical_cues.get('gaze_wandering', False))
        mouse_wandering = bool(physical_cues.get('mouse_wandering', False))

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
        
        if epistemic_struggle > 0.30:
            evidence_pool["Struggling / Stuck"].append(min(1.0, epistemic_struggle * 1.5))
        # --- INTEGRATE EPISTEMIC STATES ---
        # High confusion or frustration is a direct vote for "Struggling / Stuck"
        if confusion > 0.35:
            evidence_pool["Struggling / Stuck"].append(min(1.0, confusion * 1.2))
        if plutchik_frustration > 0.40:
            evidence_pool["Struggling / Stuck"].append(min(1.0, plutchik_frustration * 1.2))
            
        # If the user is confused, they are explicitly NOT in Deep Focus or Reading Focused
        if face_struggle > 0.25:
            evidence_pool["Reading (Focused)"].append(max(0.0, 0.4 - face_struggle))
            evidence_pool["Deep Focus"].append(max(0.0, 0.3 - face_struggle))

        sustained_same_block = False
        same_block_stuck_score = 0.0
        if current_para is not None and face_looking_at_screen and not head_turned_away and not gaze_off_screen and not gaze_wandering:
            if self.last_fixation_block is not None and current_para == self.last_fixation_block:
                sustained_same_block = True
                if gaze_x is not None and self.last_gaze_xy is not None:
                    last_x, last_y = self.last_gaze_xy
                    dx = gaze_x - last_x
                    dy = gaze_y - last_y
                    if abs(dx) <= 90 and abs(dy) <= 70:
                        self.same_block_fixation_frames += 0.45
                        if self.same_block_fixation_frames > 60:
                            self.same_block_fixation_frames += 0.10
                    else:
                        self.same_block_fixation_frames = max(0.0, self.same_block_fixation_frames - 2.0)
                else:
                    self.same_block_fixation_frames += 0.35
            else:
                self.same_block_fixation_frames = max(0.0, self.same_block_fixation_frames - 8.0)

            self.same_block_fixation_frames = min(400.0, self.same_block_fixation_frames)
            same_block_stuck_score = min(1.0, 0.02 + (self.same_block_fixation_frames / 280.0))
        else:
            self.same_block_fixation_frames = max(0.0, self.same_block_fixation_frames - 3.0)

        if sustained_same_block and current_para is not None and same_block_stuck_score > 0.30:
            # Cap pure gaze-dwell's contribution: staying on one paragraph for
            # a while is normal careful reading, not necessarily struggling.
            # Without a corroborating confusion/frustration signal, this alone
            # must never be able to peg the state at 1.0 on its own.
            evidence_pool["Struggling / Stuck"].append(min(0.55, same_block_stuck_score * 0.55))
            evidence_pool["Reading (Focused)"].append(0.04)
            evidence_pool["Deep Focus"].append(0.02)

        # Physical cues that suggest concentrated reading.
        if leaning_in or squinting:
            deep_focus_boost = 0.0
            if leaning_in:
                deep_focus_boost += 0.30
            if squinting:
                deep_focus_boost += 0.25
            evidence_pool["Deep Focus"].append(min(1.0, 0.55 + deep_focus_boost))

        if not face_looking_at_screen:
            evidence_pool["Reading (Focused)"].append(0.02)
            evidence_pool["Deep Focus"].append(0.01)
            evidence_pool["Skimming"].append(0.05)
        elif head_turned_away:
            evidence_pool["Reading (Focused)"].append(0.10)
            evidence_pool["Deep Focus"].append(0.05)
            evidence_pool["Skimming"].append(0.10)

        if self.user_profile["is_mouse_reader"]:
            if mouse_state in evidence_pool and mouse_state not in ["Inactive", "Undefined"]:
                evidence_pool[mouse_state].append(mouse_intensity)

        if head_turned_away and gaze_off_screen:
            evidence_pool["Thinking (Off-Text)"].append(0.95 if not gaze_wandering else 0.70)
        if gaze_off_screen and not gaze_wandering:
            evidence_pool["Thinking (Off-Text)"].append(0.65)
        # --- MODIFIED: Distraction Logic ---
        if mouse_wandering:
            # Mouse wandering is now the HEAVY driver for distraction
            evidence_pool["Distracted"].append(0.95)
            
        if gaze_wandering:
            # Gaze wandering is often just skimming or scanning. 
            # We drastically drop its distraction penalty and assume skimming instead.
            evidence_pool["Distracted"].append(0.05) 
            evidence_pool["Skimming"].append(0.40)
        if head_turned_away:
            evidence_pool["Distracted"].append(0.25)

        if not face_looking_at_screen and head_turned_away:
            evidence_pool["Thinking (Off-Text)"].append(0.95)
            evidence_pool["Distracted"].append(0.70)
        elif gaze_state in evidence_pool:
            if gaze_state == "Struggling / Stuck" and gaze_struggle_score > 0.65:
                evidence_pool[gaze_state].append(min(0.35, gaze_struggle_score * 0.35))
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
                # Add a bias: if epistemic states are high, use the max vote instead of average
                # This ensures one strong sign of confusion/frustration drives the state
                if state == "Struggling / Stuck":
                    instant_intensities[state] = max(votes) 
                else:
                    instant_intensities[state] = sum(votes) / len(votes)
                # If the face is struggling, Reading/Deep Focus CANNOT mathematically exceed a certain threshold
                if state in ["Reading (Focused)", "Deep Focus"] and face_struggle > 0.35:
                    # e.g., If face_struggle is 0.6, Reading is capped at 0.4
                    instant_intensities[state] = min(instant_intensities[state], 1.0 - face_struggle)

        best_smoothed_val = -1.0
        new_fused_state = self.current_fused_state
        new_fused_intensity = 0.0
        
        for state in self.smoothed_intensities.keys():
            # Use a slightly faster alpha for epistemic states if we want immediate reaction
            alpha = 0.15 if state == "Struggling / Stuck" else self.state_alpha
            self.smoothed_intensities[state] = (alpha * instant_intensities[state]) + ((1.0 - alpha) * self.smoothed_intensities[state])
            
            if self.smoothed_intensities[state] > best_smoothed_val:
                best_smoothed_val = self.smoothed_intensities[state]
                new_fused_state = state
                new_fused_intensity = self.smoothed_intensities[state]

        self.current_fused_state = new_fused_state
        fused_state = new_fused_state
        fused_intensity = new_fused_intensity

        # Only update when we actually have a paragraph: overwriting this with
        # None whenever the user looks away destroys the "last known paragraph"
        # memory that the Distracted/Thinking fallbacks below depend on.
        if current_para is not None:
            self.last_fixation_block = current_para
        self.last_gaze_xy = (gaze_x, gaze_y) if gaze_x is not None else self.last_gaze_xy

        # ---------------------------------------------------------
        # 5. TEMPORAL THRESHOLDING
        # ---------------------------------------------------------
        if (not face_looking_at_screen and (gaze_off_screen or current_para is None)) or ((gaze_wandering or mouse_wandering) and (gaze_off_screen or current_para is None)):
            self.struggle_frames = max(0, self.struggle_frames - 10)
        if fused_state == "Struggling / Stuck" or (confusion > 0.4 or frustration > 0.4):
            # Cap growth so recovery time stays bounded once the struggle
            # signal actually stops, instead of taking longer the longer it
            # ran (this had no ceiling before, so decay-to-0 could take ages).
            self.struggle_frames = min(360, self.struggle_frames + 1)
        else:
            self.struggle_frames = max(0, self.struggle_frames - 6)

        distraction_evidence = 0.0
        if fused_state in {"Thinking (Off-Text)", "Distracted"}:
            distraction_evidence += 0.55
        if gaze_state in {"Thinking (Off-Text)", "Distracted"}:
            distraction_evidence += 0.20
        if current_para is None:
            distraction_evidence += 0.15
        if mouse_state in {"Inactive", "Undefined"} and mouse_intensity < 0.1:
            distraction_evidence += 0.15
        if face_struggle < 0.25 and confusion < 0.25:
            distraction_evidence += 0.10

        self.distraction_confidence = min(1.0, max(0.0, 0.7 * self.distraction_confidence + 0.3 * distraction_evidence))

        if (not face_looking_at_screen and (gaze_off_screen or current_para is None)) or (head_turned_away and gaze_off_screen and not gaze_wandering):
            self.off_text_frames += 1
        else:
            self.off_text_frames = max(0, self.off_text_frames - 1)

        if fused_state == "Distracted" or ((gaze_wandering or mouse_wandering) and (gaze_off_screen or current_para is None)):
            self.distracted_frames += 1
        else:
            self.distracted_frames = max(0, self.distracted_frames - 1) # Drain slowly: the wandering signal is noisy frame-to-frame

        TRIGGER_THRESHOLD = 300
        DISTRACT_TRIGGER_THRESHOLD = 150
        triggered_para = None
        needs_summary = False
        
        # ---------------------------------------------------------
        # 6. PARAGRAPH DWELL LOGIC
        # ---------------------------------------------------------
        # While Distracted, current_para is usually None (the user has looked
        # away from any paragraph). Anchor the dwell timer to the last known
        # paragraph in that case instead of resetting it every frame, or the
        # fast consensus-based trigger below can never engage for distraction.
        if current_para is not None:
            dwell_para = current_para
        elif fused_state == "Distracted" and self.last_fixation_block is not None:
            dwell_para = self.last_fixation_block
        else:
            dwell_para = None

        if dwell_para != self.cur_para:
            self.cur_para = dwell_para
            self.para_t0 = now if dwell_para is not None else None
            self.para_fv = False
        elif dwell_para is not None and not self.para_fv:

            dynamic_dwell = self.dwell * 0.5 if (fused_state == "Struggling / Stuck") else self.dwell

            if now - self.para_t0 >= dynamic_dwell and now - self.para_fired.get(dwell_para, 0) >= self.cooldown:
                self.para_fired[dwell_para] = now
                self.para_fv = True
                
                # --- DECISION ENGINE (GATED BY COOLDOWN) ---
                # Replace your existing "DECISION ENGINE (GATED BY COOLDOWN)" block with this:

                if not in_cooldown:
                    # 1. Determine instantaneous intervention need for THIS frame
                    instant_need = False
                    instant_msg = ""
                    instant_trigger_para = None
                    instant_needs_sum = False

                    if plutchik_frustration > self.config["frust_thresh"] and face_struggle > self.config["face_struggle_thresh"]:
                        instant_need = True
                        instant_msg = "INTERVENTION: High Frustration. Take a short break, don't force it."
                    elif fused_state == "Thinking (Off-Text)":
                        instant_need = True
                        instant_msg = "INTERVENTION: Taking time to process. Let me know if you need help."
                    elif fused_state == "Distracted":
                        instant_need = True
                        # Distraction is a nudge, not a content request: don't
                        # fetch/generate a paragraph summary for it.
                        instant_trigger_para = None
                        instant_needs_sum = False
                        instant_msg = "INTERVENTION: You seem distracted. Try to refocus on the page."
                    elif fused_state == "Struggling / Stuck":
                        instant_need = True
                        instant_trigger_para = current_para
                        instant_needs_sum = True
                        instant_msg = "INTERVENTION: Cognitive load detected. Generating paragraph summary..."

                    # 2. Append to rolling buffer
                    self.intervention_buffer.append({
                        "needed": instant_need,
                        "msg": instant_msg,
                        "para": instant_trigger_para,
                        "summary": instant_needs_sum
                    })

                    # 3. Evaluate Consensus (Has the user been struggling for the majority of the last second?)
                    positive_frames = [frame for frame in self.intervention_buffer if frame["needed"]]
                    
                    if len(positive_frames) >= self.config["consensus_required"]:
                        self.last_tip_time = now 
                        self.help_active = True
                        
                        # Trigger based on the most common state in the positive frames
                        # (e.g., if they were mostly frustrated, show the frustration message)
                        counts = {}
                        for frame in positive_frames:
                            counts[frame["msg"]] = counts.get(frame["msg"], 0) + 1
                            
                        best_match = max(positive_frames, key=lambda f: counts[f["msg"]])
                        
                        self.system_message = best_match["msg"]
                        triggered_para = best_match["para"]
                        needs_summary = best_match["summary"]
                        
                        # Flush the buffer after a successful trigger to prevent double-firing
                        self.intervention_buffer.clear()
        # ---------------------------------------------------------
        # 7. GLOBAL STATE INTERVENTIONS
        # ---------------------------------------------------------
        # Pick whichever condition is dominant *right now* and always reflect
        # it in the message. Previously this whole block was gated behind
        # 'not self.help_active', so once one reason fired (e.g. struggling)
        # its message was frozen until a full reset - a newer, more relevant
        # reason (e.g. looking away -> distracted) could never take over.
        if self.distracted_frames > DISTRACT_TRIGGER_THRESHOLD:
            desired_reason = "distracted"
        elif fused_state == "Thinking (Off-Text)":
            desired_reason = "thinking"
        elif self.struggle_frames > TRIGGER_THRESHOLD:
            desired_reason = "struggling"
        else:
            desired_reason = None

        if desired_reason is not None:
            reason_just_started = desired_reason != self.active_reason
            self.help_active = True
            self.active_reason = desired_reason

            if desired_reason == "distracted":
                # Distraction is a nudge, not a content request: don't
                # fetch/generate a paragraph or whole-paper summary for it.
                self.system_message = "INTERVENTION: You seem distracted. Try to refocus on the page."
            elif desired_reason == "thinking":
                self.system_message = "INTERVENTION: Taking time to process. Let me know if you need help."
            else:
                if plutchik_frustration > 0.6:
                    self.system_message = "INTERVENTION: High Frustration detected. Generating paragraph summary..."
                else:
                    self.system_message = "INTERVENTION: Cognitive load detected. Generating paragraph summary..."

                # Only (re)generate the paragraph summary once per new
                # occurrence of struggling, gated by cooldown, so we don't
                # keep re-requesting it every frame while it stays elevated.
                if reason_just_started and not in_cooldown:
                    self.last_tip_time = now
                    triggered_para = current_para
                    needs_summary = True
        elif self.help_active and not needs_summary:
            self.help_active = False
            self.active_reason = None
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
            "profile": self.user_profile,

        }

    def reset_intervention(self):
        self.struggle_frames = 0
        self.distracted_frames = 0
        self.help_active = False
        self.active_reason = None
        self.system_message = "Listening to User State..."
        self.last_tip_time = time.time()
        self.reset_dwell()
        
    def reset_dwell(self):
        self.cur_para = None
        self.para_t0 = None
        self.para_fv = False