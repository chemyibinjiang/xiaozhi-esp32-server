import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


fake_util_module = types.ModuleType("core.utils.util")


async def _fake_audio_to_data(*args, **kwargs):
    return []


fake_util_module.audio_to_data = _fake_audio_to_data
sys.modules.setdefault("core.utils.util", fake_util_module)


fake_rate_controller_module = types.ModuleType("core.utils.audioRateController")


class _DummyAudioRateController:
    def __init__(self, *args, **kwargs):
        self.queue = []
        self.queue_empty_event = None


fake_rate_controller_module.AudioRateController = _DummyAudioRateController
sys.modules.setdefault("core.utils.audioRateController", fake_rate_controller_module)


from core.handle.sendAudioHandle import _resolve_tts_stop_buffer_ms


class SendAudioTimingTest(unittest.TestCase):
    def test_zero_requested_extra_keeps_minimum_stop_buffer(self):
        conn = SimpleNamespace(
            config={
                "tts_stop_extra_buffer_ms": 0,
                "tts_stop_min_buffer_ms": 240,
            }
        )

        requested, minimum, effective = _resolve_tts_stop_buffer_ms(conn, 60)

        self.assertEqual(0, requested)
        self.assertEqual(240, minimum)
        self.assertEqual(240, effective)

    def test_requested_extra_above_floor_is_preserved(self):
        conn = SimpleNamespace(
            config={
                "tts_stop_extra_buffer_ms": 420,
                "tts_stop_min_buffer_ms": 240,
            }
        )

        requested, minimum, effective = _resolve_tts_stop_buffer_ms(conn, 60)

        self.assertEqual(420, requested)
        self.assertEqual(240, minimum)
        self.assertEqual(420, effective)


if __name__ == "__main__":
    unittest.main()
