import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config.safe_rotating_file_sink import SafeRotatingFileSink


class SafeRotatingFileSinkTest(unittest.TestCase):
    def test_rotation_creates_archived_log_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "server.log"
            sink = SafeRotatingFileSink(log_path, rotation=16, retention=None)
            try:
                sink.write("1234567890\n")
                sink.write("abcdefghij\n")

                rotated_files = list(Path(temp_dir).glob("server.*.log"))
                self.assertEqual(1, len(rotated_files))
                self.assertEqual(
                    "1234567890\n", rotated_files[0].read_text(encoding="utf-8")
                )
                self.assertEqual("abcdefghij\n", log_path.read_text(encoding="utf-8"))
            finally:
                sink.stop()

    def test_permission_error_during_rotation_keeps_logging(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "server.log"
            sink = SafeRotatingFileSink(
                log_path,
                rotation=16,
                retention=None,
                rotation_retry_interval=0,
            )
            try:
                sink.write("1234567890\n")

                original_replace = sys.modules[
                    "config.safe_rotating_file_sink"
                ].os.replace
                replace_calls = {"count": 0}

                def flaky_replace(src, dst):
                    replace_calls["count"] += 1
                    if replace_calls["count"] == 1:
                        raise PermissionError("locked")
                    return original_replace(src, dst)

                with mock.patch(
                    "config.safe_rotating_file_sink.os.replace",
                    side_effect=flaky_replace,
                ):
                    sink.write("abcdefghij\n")
                    sink.write("klmnopqrst\n")

                rotated_files = list(Path(temp_dir).glob("server.*.log"))
                self.assertEqual(1, len(rotated_files))
                self.assertEqual(
                    "1234567890\nabcdefghij\n",
                    rotated_files[0].read_text(encoding="utf-8"),
                )
                self.assertEqual("klmnopqrst\n", log_path.read_text(encoding="utf-8"))
            finally:
                sink.stop()


if __name__ == "__main__":
    unittest.main()
