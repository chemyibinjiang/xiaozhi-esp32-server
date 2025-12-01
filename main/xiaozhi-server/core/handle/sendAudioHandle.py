import json
import time
import asyncio
from core.utils import textUtils
from core.utils.util import audio_to_data
from core.providers.tts.dto.dto import SentenceType
from core.utils.audioRateController import AudioRateController

TAG = __name__


async def sendAudioMessage(conn, sentenceType, audios, text):
    if conn.tts.tts_audio_first_sentence:
        conn.logger.bind(tag=TAG).info(f"发送第一段语音: {text}")
        conn.tts.tts_audio_first_sentence = False
        await send_tts_message(conn, "start", None)

    if sentenceType == SentenceType.FIRST:
        await send_tts_message(conn, "sentence_start", text)

    await sendAudio(conn, audios)
    # 发送句子开始消息
    if sentenceType is not SentenceType.MIDDLE:
        conn.logger.bind(tag=TAG).info(f"发送音频消息: {sentenceType}, {text}")

    # 发送结束消息（如果是最后一个文本）
    if sentenceType == SentenceType.LAST:
        await send_tts_message(conn, "stop", None)
        conn.client_is_speaking = False
        if conn.close_after_chat:
            await conn.close()


async def _send_to_mqtt_gateway(conn, opus_packet, timestamp, sequence):
    """
    发送带16字节头部的opus数据包给mqtt_gateway
    Args:
        conn: 连接对象
        opus_packet: opus数据包
        timestamp: 时间戳
        sequence: 序列号
    """
    # 为opus数据包添加16字节头部
    header = bytearray(16)
    header[0] = 1  # type
    header[2:4] = len(opus_packet).to_bytes(2, "big")  # payload length
    header[4:8] = sequence.to_bytes(4, "big")  # sequence
    header[8:12] = timestamp.to_bytes(4, "big")  # 时间戳
    header[12:16] = len(opus_packet).to_bytes(4, "big")  # opus长度

    # 发送包含头部的完整数据包
    complete_packet = bytes(header) + opus_packet
    await conn.websocket.send(complete_packet)


# 播放音频 - 使用 AudioRateController 进行精确流控
async def sendAudio(conn, audios, frame_duration=60):
    """
    发送音频包，使用 AudioRateController 进行精确的流量控制

    Args:
        conn: 连接对象
        audios: 单个opus包(bytes) 或 opus包列表
        frame_duration: 帧时长（毫秒），默认60ms

    改进点：
    1. 使用单一时间基准，避免累积误差
    2. 每次检查队列时重新计算 elapsed_ms，更精准
    3. 支持高并发而不产生时间偏差
    """
    if audios is None or len(audios) == 0:
        return

    # 获取发送延迟配置
    send_delay = conn.config.get("tts_audio_send_delay", -1) / 1000.0

    if isinstance(audios, bytes):
        # 单个 opus 包处理
        await _sendAudio_single(conn, audios, send_delay, frame_duration)
    else:
        # 音频列表处理（如文件型音频）
        await _sendAudio_list(conn, audios, send_delay, frame_duration)


async def _sendAudio_single(conn, opus_packet, send_delay, frame_duration=60):
    """
    发送单个 opus 包
    使用 AudioRateController 进行流控
    """
    # 重置流控状态，第一次读取和会话发生转变时
    if not hasattr(conn, "audio_rate_controller") or conn.audio_flow_control.get("sentence_id") != conn.sentence_id:
        if hasattr(conn, "audio_rate_controller"):
            conn.audio_rate_controller.reset()
        else:
            conn.audio_rate_controller = AudioRateController(frame_duration)
            conn.audio_rate_controller.reset()

        conn.audio_flow_control = {
            "packet_count": 0,
            "sequence": 0,
            "sentence_id": conn.sentence_id,
        }

    if conn.client_abort:
        return

    conn.last_activity_time = time.time() * 1000

    rate_controller = conn.audio_rate_controller
    flow_control = conn.audio_flow_control
    packet_count = flow_control["packet_count"]

    # 预缓冲：前5个包直接发送，不做延迟
    pre_buffer_count = 5

    if packet_count < pre_buffer_count or send_delay > 0:
        # 预缓冲阶段或固定延迟模式，直接发送
        await _do_send_audio(conn, opus_packet, flow_control, frame_duration)

        if send_delay > 0 and packet_count >= pre_buffer_count:
            await asyncio.sleep(send_delay)
    else:
        # 使用流控器进行精确的速率控制
        rate_controller.add_audio(opus_packet)

        async def send_callback(packet):
            await _do_send_audio(conn, packet, flow_control, frame_duration)

        await rate_controller.check_queue(send_callback)

    # 更新流控状态
    flow_control["packet_count"] += 1
    flow_control["sequence"] += 1


async def _sendAudio_list(conn, audios, send_delay, frame_duration=60):
    """
    发送音频列表（如文件型音频）
    """
    if not audios:
        return

    rate_controller = AudioRateController(frame_duration)
    rate_controller.reset()

    flow_control = {
        "packet_count": 0,
        "sequence": 0,
    }

    # 预缓冲：前5个包直接发送
    pre_buffer_frames = min(5, len(audios))
    for i in range(pre_buffer_frames):
        if conn.client_abort:
            return
        await _do_send_audio(conn, audios[i], flow_control, frame_duration)
        conn.client_is_speaking = True

    remaining_audios = audios[pre_buffer_frames:]

    # 处理剩余音频帧
    for i, opus_packet in enumerate(remaining_audios):
        if conn.client_abort:
            break

        conn.last_activity_time = time.time() * 1000

        if send_delay > 0:
            # 固定延迟模式
            await asyncio.sleep(send_delay)
        else:
            # 使用流控器进行精确延迟
            rate_controller.add_audio(opus_packet)

            async def send_callback(packet):
                await _do_send_audio(conn, packet, flow_control, frame_duration)

            await rate_controller.check_queue(send_callback)
            conn.client_is_speaking = True
            continue

        await _do_send_audio(conn, opus_packet, flow_control, frame_duration)
        conn.client_is_speaking = True


async def _do_send_audio(conn, opus_packet, flow_control, frame_duration=60):
    """
    执行实际的音频发送
    """
    packet_index = flow_control.get("packet_count", 0)
    sequence = flow_control.get("sequence", 0)

    if conn.conn_from_mqtt_gateway:
        # 计算时间戳（基于播放位置）
        start_time = time.time()
        timestamp = int(start_time * 1000) % (2**32)
        await _send_to_mqtt_gateway(conn, opus_packet, timestamp, sequence)
    else:
        # 直接发送opus数据包
        await conn.websocket.send(opus_packet)

    # 更新流控状态
    flow_control["packet_count"] = packet_index + 1
    flow_control["sequence"] = sequence + 1


async def send_tts_message(conn, state, text=None):
    """发送 TTS 状态消息"""
    if text is None and state == "sentence_start":
        return
    message = {"type": "tts", "state": state, "session_id": conn.session_id}
    if text is not None:
        message["text"] = textUtils.check_emoji(text)

    # TTS播放结束
    if state == "stop":
        # 播放提示音
        tts_notify = conn.config.get("enable_stop_tts_notify", False)
        if tts_notify:
            stop_tts_notify_voice = conn.config.get(
                "stop_tts_notify_voice", "config/assets/tts_notify.mp3"
            )
            audios = audio_to_data(stop_tts_notify_voice, is_opus=True)
            await sendAudio(conn, audios)
        # 清除服务端讲话状态
        conn.clearSpeakStatus()

    # 发送消息到客户端
    await conn.websocket.send(json.dumps(message))


async def send_stt_message(conn, text):
    """发送 STT 状态消息"""
    end_prompt_str = conn.config.get("end_prompt", {}).get("prompt")
    if end_prompt_str and end_prompt_str == text:
        await send_tts_message(conn, "start")
        return

    # 解析JSON格式，提取实际的用户说话内容
    display_text = text
    try:
        # 尝试解析JSON格式
        if text.strip().startswith("{") and text.strip().endswith("}"):
            parsed_data = json.loads(text)
            if isinstance(parsed_data, dict) and "content" in parsed_data:
                # 如果是包含说话人信息的JSON格式，只显示content部分
                display_text = parsed_data["content"]
                # 保存说话人信息到conn对象
                if "speaker" in parsed_data:
                    conn.current_speaker = parsed_data["speaker"]
    except (json.JSONDecodeError, TypeError):
        # 如果不是JSON格式，直接使用原始文本
        display_text = text
    stt_text = textUtils.get_string_no_punctuation_or_emoji(display_text)
    await conn.websocket.send(
        json.dumps({"type": "stt", "text": stt_text, "session_id": conn.session_id})
    )
    await send_tts_message(conn, "start")
