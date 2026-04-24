import json

TAG = __name__


def _should_ignore_abort(conn, reason):
    reason = str(reason or "").strip()
    if reason != "wake_word_detected":
        return False
    if not conn.config.get("disable_listen_during_chat", False):
        return False
    return bool(
        conn.client_is_speaking or not conn.llm_finish_task or conn.has_external_busy()
    )


async def handleAbortMessage(conn, reason=None):
    if _should_ignore_abort(conn, reason):
        conn.logger.bind(tag=TAG).info(
            f"Ignore abort during chat: reason={reason}, "
            f"client_is_speaking={conn.client_is_speaking}, "
            f"llm_finish_task={conn.llm_finish_task}"
        )
        return

    conn.logger.bind(tag=TAG).info(f"Abort message received, reason={reason}")
    conn.client_abort = True
    conn.clear_queues()
    await conn.websocket.send(
        json.dumps({"type": "tts", "state": "stop", "session_id": conn.session_id})
    )
    conn.clearSpeakStatus()
    conn.logger.bind(tag=TAG).info("Abort message received-end")
