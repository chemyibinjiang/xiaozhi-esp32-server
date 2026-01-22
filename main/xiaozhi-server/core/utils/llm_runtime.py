import threading

_abort_blocked = {}
_lock = threading.Lock()


def block_abort(session_id: str) -> None:
    if not session_id:
        return
    with _lock:
        _abort_blocked[session_id] = True


def unblock_abort(session_id: str) -> None:
    if not session_id:
        return
    with _lock:
        _abort_blocked.pop(session_id, None)


def is_abort_blocked(session_id: str) -> bool:
    if not session_id:
        return False
    with _lock:
        return _abort_blocked.get(session_id, False)


def clear_session(session_id: str) -> None:
    unblock_abort(session_id)
