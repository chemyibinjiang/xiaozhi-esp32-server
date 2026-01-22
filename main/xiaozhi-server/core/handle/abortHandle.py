import json

from core.utils.llm_runtime import is_abort_blocked

TAG = __name__


async def handleAbortMessage(conn):
    if is_abort_blocked(conn.session_id):
        conn.logger.bind(tag=TAG).info("Abort blocked for active LLM status output")
        return
    conn.logger.bind(tag=TAG).info("Abort message received")
    # 设置成打断状态，会自动打断llm、tts任务
    conn.client_abort = True
    conn.clear_queues()
    # 打断客户端说话状态
    await conn.websocket.send(
        json.dumps({"type": "tts", "state": "stop", "session_id": conn.session_id})
    )
    conn.clearSpeakStatus()
    conn.logger.bind(tag=TAG).info("Abort message received-end")
