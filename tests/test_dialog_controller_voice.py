import time
import unittest

from dialog_controller import DialogController


class FakeVoiceAssistant:
    def __init__(self):
        self.prompts = []

    def speak(self, text):
        self.prompts.append(text)


class DialogControllerVoiceTests(unittest.TestCase):
    def test_downgrades_reading_when_face_is_not_looking_at_screen(self):
        controller = DialogController(
            dwell=0.05,
            cooldown=0.05,
            calibration_duration=0.01,
            tip_cooldown=0.01,
            voice_assistant=None,
        )
        controller.is_calibrating = False
        controller.start_time = time.time()
        controller.state_alpha = 1.0

        result = controller.evaluate(
            mouse_data={"state": "Reading (Focused)", "intensity": 0.2},
            gaze_data={"state": "Reading (Focused)", "score": 0.0, "y": 540},
            epistemic_state={},
            emotion_scores={},
            current_para=None,
            physical_cues={"face_looking_at_screen": False},
        )

        self.assertNotIn(result["fused_state"], {"Reading (Focused)", "Deep Focus"})

    def test_speaks_after_sustained_off_text_state(self):
        voice_assistant = FakeVoiceAssistant()
        controller = DialogController(
            dwell=0.05,
            cooldown=0.05,
            calibration_duration=0.01,
            tip_cooldown=0.01,
            voice_assistant=voice_assistant,
        )
        controller.is_calibrating = False
        controller.start_time = time.time()
        controller.state_alpha = 1.0
        controller.voice_prompt_cooldown = 0.0

        for _ in range(20):
            controller.evaluate(
                mouse_data={"state": "Inactive", "intensity": 0.0},
                gaze_data={"state": "Thinking (Off-Text)", "score": 0.8, "y": 540},
                epistemic_state={},
                emotion_scores={},
                current_para=None,
                physical_cues={"face_looking_at_screen": False, "gaze_off_screen": True},
            )

        self.assertGreaterEqual(len(voice_assistant.prompts), 1)

    def test_builds_stuck_state_after_sustained_same_block_fixation(self):
        controller = DialogController(
            dwell=0.05,
            cooldown=0.05,
            calibration_duration=0.01,
            tip_cooldown=0.01,
            voice_assistant=None,
        )
        controller.is_calibrating = False
        controller.start_time = time.time()
        controller.state_alpha = 1.0

        result = None
        for _ in range(80):
            result = controller.evaluate(
                mouse_data={"state": "Reading (Focused)", "intensity": 0.2},
                gaze_data={"state": "Reading (Focused)", "score": 0.1, "y": 540},
                epistemic_state={},
                emotion_scores={},
                current_para=3,
                physical_cues={"face_looking_at_screen": True, "head_turned_away": False, "gaze_off_screen": False, "gaze_wandering": False, "mouse_wandering": False},
            )

        self.assertGreaterEqual(result["fused_intensity"], 0.15)
        self.assertEqual(result["fused_state"], "Struggling / Stuck")


if __name__ == "__main__":
    unittest.main()
