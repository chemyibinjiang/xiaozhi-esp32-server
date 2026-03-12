import sys
import types
import unittest
from pathlib import Path
from types import MethodType


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


class _FakeLogger:
    def bind(self, **kwargs):
        return self

    def info(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


fake_logger_module = types.ModuleType("config.logger")
fake_logger_module.setup_logging = lambda: _FakeLogger()
sys.modules.setdefault("config.logger", fake_logger_module)

from core.providers.llm.codex.codex import _CodexSession


class CodexPromptStateTest(unittest.TestCase):
    def _make_session(self) -> _CodexSession:
        session = _CodexSession(
            {
                "codex_bin": "codex.cmd",
                "model_name": "gpt-5.3-codex",
                "workspace": str(SCRIPT_DIR),
                "system_prompt_mode": "first_turn",
                "bootstrap_mode": "none",
                "auto_approve": True,
                "network_access": True,
                "export_api_key": False,
            },
            "test-session",
        )
        session._captured_prompts = []
        session._fake_started = False

        def fake_start(self):
            if self._fake_started:
                return
            self._fake_started = True
            self.proc = object()
            self.thread_id = "thread-1"
            self._system_prompt_sent = False
            self._bootstrap_history = True

        def fake_stream_turn(self, prompt_text, emit_events, user_text=None, **kwargs):
            self.start()
            self._captured_prompts.append(
                {
                    "prompt_text": prompt_text,
                    "user_text": user_text,
                    "system_prompt_sent": self._system_prompt_sent,
                }
            )
            yield "ok"

        session.start = MethodType(fake_start, session)
        session._stream_turn = MethodType(fake_stream_turn, session)
        return session

    def test_first_turn_only_sends_system_prompt_once(self):
        session = self._make_session()

        first_dialogue = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hello"},
        ]
        list(session.stream_response(first_dialogue))

        self.assertTrue(session._system_prompt_sent)
        self.assertEqual("SYS\n\nhello", session._captured_prompts[0]["prompt_text"])
        self.assertTrue(session._captured_prompts[0]["system_prompt_sent"])

        second_dialogue = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "again"},
        ]
        list(session.stream_response(second_dialogue))

        self.assertEqual("again", session._captured_prompts[1]["prompt_text"])
        self.assertEqual("again", session._captured_prompts[1]["user_text"])


if __name__ == "__main__":
    unittest.main()
