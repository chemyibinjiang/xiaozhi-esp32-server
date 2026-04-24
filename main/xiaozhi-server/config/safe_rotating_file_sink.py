import os
import re
import time
from datetime import datetime
from pathlib import Path
from threading import RLock


_SIZE_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?b)?\s*$", re.IGNORECASE)
_RETENTION_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(second|seconds|minute|minutes|hour|hours|day|days)?\s*$",
    re.IGNORECASE,
)
_SIZE_UNITS = {
    None: 1,
    "b": 1,
    "kb": 1024,
    "mb": 1024 * 1024,
    "gb": 1024 * 1024 * 1024,
    "tb": 1024 * 1024 * 1024 * 1024,
}
_RETENTION_UNITS = {
    None: 24 * 60 * 60,
    "second": 1,
    "seconds": 1,
    "minute": 60,
    "minutes": 60,
    "hour": 60 * 60,
    "hours": 60 * 60,
    "day": 24 * 60 * 60,
    "days": 24 * 60 * 60,
}


def parse_rotation_bytes(value):
    if value in (None, "", 0):
        return None
    if isinstance(value, (int, float)):
        return max(int(value), 0) or None
    if not isinstance(value, str):
        raise ValueError(f"Unsupported rotation value: {value!r}")

    match = _SIZE_PATTERN.match(value)
    if not match:
        raise ValueError(f"Unsupported rotation value: {value!r}")

    number, unit = match.groups()
    return int(float(number) * _SIZE_UNITS[unit.lower() if unit else None])


def parse_retention_seconds(value):
    if value in (None, "", 0):
        return None
    if isinstance(value, (int, float)):
        return max(int(value), 0) or None
    if not isinstance(value, str):
        raise ValueError(f"Unsupported retention value: {value!r}")

    match = _RETENTION_PATTERN.match(value)
    if not match:
        raise ValueError(f"Unsupported retention value: {value!r}")

    number, unit = match.groups()
    return int(float(number) * _RETENTION_UNITS[unit.lower() if unit else None])


class SafeRotatingFileSink:
    def __init__(
        self,
        path,
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8",
        rotation_retry_interval=5.0,
    ):
        self._path = os.path.abspath(path)
        self._encoding = encoding
        self._rotation_bytes = parse_rotation_bytes(rotation)
        self._retention_seconds = parse_retention_seconds(retention)
        self._rotation_retry_interval = max(float(rotation_retry_interval), 0.0)
        self._next_rotation_retry_at = 0.0
        self._lock = RLock()
        self._file = None
        self._size = 0

        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._open_file()
        self._cleanup_old_files()

    def write(self, message):
        text = str(message)
        payload = text.encode(self._encoding, errors="replace")

        with self._lock:
            if self._should_rotate(len(payload)):
                self._rotate_if_needed()

            self._file.write(text)
            self._file.flush()
            self._size += len(payload)

    def stop(self):
        with self._lock:
            if self._file and not self._file.closed:
                self._file.flush()
                self._file.close()

    close = stop

    def _should_rotate(self, incoming_size):
        if not self._rotation_bytes:
            return False
        if self._size <= 0:
            return False
        if (self._size + incoming_size) <= self._rotation_bytes:
            return False
        return time.monotonic() >= self._next_rotation_retry_at

    def _rotate_if_needed(self):
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

        rotated_path = self._build_rotated_path()

        try:
            if os.path.exists(self._path):
                os.replace(self._path, rotated_path)
        except PermissionError:
            self._next_rotation_retry_at = (
                time.monotonic() + self._rotation_retry_interval
            )
            self._open_file()
            return
        except FileNotFoundError:
            pass

        self._next_rotation_retry_at = 0.0
        self._open_file()
        self._cleanup_old_files()

    def _open_file(self):
        self._file = open(self._path, "a", encoding=self._encoding, buffering=1)
        self._size = os.path.getsize(self._path) if os.path.exists(self._path) else 0

    def _build_rotated_path(self):
        path = Path(self._path)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        if path.suffix:
            return str(path.with_name(f"{path.stem}.{timestamp}{path.suffix}"))
        return str(path.with_name(f"{path.name}.{timestamp}"))

    def _cleanup_old_files(self):
        if not self._retention_seconds:
            return

        cutoff = time.time() - self._retention_seconds
        for old_file in self._iter_rotated_files():
            try:
                if old_file.stat().st_mtime < cutoff:
                    old_file.unlink()
            except OSError:
                continue

    def _iter_rotated_files(self):
        current_path = Path(self._path)
        if current_path.suffix:
            pattern = f"{current_path.stem}.*{current_path.suffix}"
        else:
            pattern = f"{current_path.name}.*"

        for candidate in current_path.parent.glob(pattern):
            if candidate.resolve() != current_path.resolve():
                yield candidate
