import json
import uuid
import asyncio
from core.utils.dialogue import Message
from core.providers.tts.dto.dto import ContentType
from core.handle.helloHandle import checkWakeupWords
from plugins_func.register import Action, ActionResponse
from core.handle.sendAudioHandle import send_stt_message
from core.utils import textUtils
from core.utils.util import remove_punctuation_and_length, sanitize_tool_name
from core.providers.tts.dto.dto import TTSMessageDTO, SentenceType
from core.providers.tools.device_mcp import call_mcp_tool

TAG = __name__


async def handle_user_intent(conn, text):
    # 预处理输入文本，处理可能的JSON格式
    try:
        if text.strip().startswith('{') and text.strip().endswith('}'):
            parsed_data = json.loads(text)
            if isinstance(parsed_data, dict) and "content" in parsed_data:
                text = parsed_data["content"]  # 提取content用于意图分析
                conn.current_speaker = parsed_data.get("speaker")  # 保留说话人信息
    except (json.JSONDecodeError, TypeError):
        pass

    # 检查是否有明确的退出命令
    _, filtered_text = remove_punctuation_and_length(text)
    if await check_direct_exit(conn, filtered_text):
        return True

    # 检查是否是唤醒词
    if await checkWakeupWords(conn, filtered_text):
        return True

    if await handle_pending_direct_photo_confirmation(conn, text, filtered_text):
        return True

    _update_server_photo_confirmation_state(conn, filtered_text)

    # Fast path: photo-navigation commands go directly to MCP (no Codex).
    if await handle_direct_photo_navigation_intent(conn, text, filtered_text):
        return True

    # Fast path: take-photo commands go directly to MCP (no Codex).
    if await handle_direct_photo_intent(conn, text, filtered_text):
        return True

    if conn.intent_type == "function_call":
        # 使用支持function calling的聊天方法,不再进行意图分析
        return False
    # 使用LLM进行意图分析
    intent_result = await analyze_intent_with_llm(conn, text)
    if not intent_result:
        return False
    # 会话开始时生成sentence_id
    conn.sentence_id = str(uuid.uuid4().hex)
    # 处理各种意图
    return await process_intent_result(conn, intent_result, text)


async def check_direct_exit(conn, text):
    """检查是否有明确的退出命令"""
    _, text = remove_punctuation_and_length(text)
    cmd_exit = conn.cmd_exit
    for cmd in cmd_exit:
        if text == cmd:
            conn.logger.bind(tag=TAG).info(f"识别到明确的退出命令: {text}")
            await send_stt_message(conn, text)
            await conn.close()
            return True
    return False


async def analyze_intent_with_llm(conn, text):
    """使用LLM分析用户意图"""
    if not hasattr(conn, "intent") or not conn.intent:
        conn.logger.bind(tag=TAG).warning("意图识别服务未初始化")
        return None

    # 对话历史记录
    dialogue = conn.dialogue
    try:
        intent_result = await conn.intent.detect_intent(conn, dialogue.dialogue, text)
        return intent_result
    except Exception as e:
        conn.logger.bind(tag=TAG).error(f"意图识别失败: {str(e)}")

    return None


def _normalize_text_for_match(text: str) -> str:
    return (text or "").strip().lower().replace(" ", "")


def _contains_any(text: str, words) -> bool:
    return any(w in text for w in words)


def _starts_with_any(text: str, words) -> bool:
    return any(text.startswith(w) for w in words)


def _looks_like_question_reply(text: str) -> bool:
    if not text:
        return False
    if text.endswith(("吗", "么", "嘛", "呢")):
        return True
    question_tokens = (
        "可不可以",
        "能不能",
        "行不行",
        "要不要",
        "是不是",
        "为什么",
        "怎么",
        "如何",
    )
    return _contains_any(text, question_tokens)


def _is_direct_photo_command(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return False

    block_keywords = [
        "\u4e3a\u4ec0\u4e48",
        "\u539f\u7406",
        "\u6b65\u9aa4",
        "\u6ce8\u610f\u4e8b\u9879",
        "\u600e\u4e48\u505a",
        "\u5982\u4f55\u505a",
        "\u4ec0\u4e48\u610f\u601d",
    ]
    if _contains_any(norm, block_keywords):
        return False

    trigger_keywords = [
        "\u62cd\u7167",
        "\u62cd\u4e00\u5f20",
        "\u62cd\u4e2a\u7167",
        "\u62cd\u5f20\u7167",
        "\u62cd\u5f20\u7167\u7247",
        "\u62cd\u4e00\u5f20\u7167\u7247",
        "\u7167\u4e00\u4e0b",
        "\u770b\u4e00\u4e0b\u524d\u9762",
        "\u770b\u770b\u524d\u9762",
        "\u770b\u770b\u5f53\u524d\u753b\u9762",
        "\u770b\u4e00\u4e0b\u5f53\u524d\u753b\u9762",
        "\u5e2e\u6211\u770b\u4e00\u4e0b",
        "\u5e2e\u6211\u770b\u4e00\u773c",
        "\u770b\u4e00\u773c\u524d\u9762",
    ]
    return _contains_any(norm, trigger_keywords)


def _classify_photo_nav_command(filtered_text: str) -> str:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return ""

    block_keywords = [
        "\u4e3a\u4ec0\u4e48",
        "\u539f\u7406",
        "\u6b65\u9aa4",
        "\u6ce8\u610f\u4e8b\u9879",
        "\u600e\u4e48\u505a",
        "\u5982\u4f55\u505a",
        "\u4ec0\u4e48\u610f\u601d",
    ]
    if _contains_any(norm, block_keywords):
        return ""

    previous_keywords = [
        "\u4e0a\u4e00\u5f20",
        "\u524d\u4e00\u5f20",
        "\u4e0a\u4e00\u5f20\u7167\u7247",
        "\u524d\u4e00\u5f20\u7167\u7247",
        "\u4e0a\u5f20",
        "\u524d\u5f20",
    ]
    if _contains_any(norm, previous_keywords):
        return "previous"

    latest_keywords = [
        "\u67e5\u770b\u6700\u8fd1\u7167\u7247",
        "\u770b\u6700\u8fd1\u7167\u7247",
        "\u770b\u770b\u6700\u8fd1\u7167\u7247",
        "\u6700\u8fd1\u7167\u7247",
        "\u6700\u65b0\u7167\u7247",
        "\u6700\u8fd1\u4e00\u5f20",
        "\u6700\u65b0\u4e00\u5f20",
    ]
    if _contains_any(norm, latest_keywords):
        return "latest"

    return ""


def _build_direct_photo_question(original_text: str, default_question: str) -> str:
    text = (original_text or "").strip()
    if not text:
        return default_question
    if _contains_any(
        text,
        [
            "\u5206\u6790",
            "\u8bc6\u522b",
            "\u63cf\u8ff0",
            "\u770b\u770b",
            "\u770b\u4e00\u4e0b",
            "\u5e2e\u6211\u770b",
        ],
    ):
        return text
    return default_question


def _to_plain_data(payload):
    if payload is None:
        return None
    if isinstance(payload, (dict, list, str, int, float, bool)):
        return payload
    if hasattr(payload, "model_dump"):
        try:
            return payload.model_dump()
        except Exception:
            pass
    if hasattr(payload, "__dict__"):
        try:
            return dict(payload.__dict__)
        except Exception:
            pass
    return str(payload)


def _try_parse_json_text(text: str):
    if not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw:
        return None
    if not (raw.startswith("{") or raw.startswith("[")):
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _extract_server_mcp_payload(raw_result):
    data = _to_plain_data(raw_result)
    if isinstance(data, str):
        parsed = _try_parse_json_text(data)
        return parsed if parsed is not None else data

    if isinstance(data, dict):
        content = data.get("content")
        if isinstance(content, list):
            for item in content:
                item_data = _to_plain_data(item)
                if isinstance(item_data, dict):
                    text = item_data.get("text")
                    if isinstance(text, str):
                        parsed_text = _try_parse_json_text(text)
                        if parsed_text is not None:
                            return parsed_text
                        if text.strip():
                            return text.strip()
        return data

    if isinstance(data, list):
        for item in data:
            extracted = _extract_server_mcp_payload(item)
            if extracted is not None:
                return extracted
    return data


def _extract_text_from_result_payload(payload):
    data = _to_plain_data(payload)

    if isinstance(data, str):
        text = data.strip()
        if text and not text.startswith("{") and not text.startswith("["):
            return text
        parsed = _try_parse_json_text(text)
        if parsed is not None:
            return _extract_text_from_result_payload(parsed)
        return ""

    if isinstance(data, dict):
        for key in ["response", "message", "text", "description"]:
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        nested = data.get("result")
        if nested is not None:
            nested_text = _extract_text_from_result_payload(nested)
            if nested_text:
                return nested_text

        photo_meta = data.get("photo_meta")
        if isinstance(photo_meta, dict):
            file_name = str(photo_meta.get("file_name", "")).strip()
            if file_name:
                return f"\u62cd\u597d\u4e86\uff0c\u5df2\u4fdd\u5b58\u4e3a {file_name}"
        return ""

    if isinstance(data, list):
        for item in data:
            text = _extract_text_from_result_payload(item)
            if text:
                return text
    return ""


def _extract_direct_photo_reply(raw_result) -> str:
    payload = _extract_server_mcp_payload(raw_result)
    text = _extract_text_from_result_payload(payload)
    if text:
        return text
    return ""


def _get_server_mcp_manager(conn):
    func_handler = getattr(conn, "func_handler", None)
    if not func_handler:
        return None
    server_executor = getattr(func_handler, "server_mcp_executor", None)
    if not server_executor:
        return None
    return getattr(server_executor, "mcp_manager", None)


async def _execute_server_mcp_tool_direct(conn, tool_name: str, arguments: dict):
    manager = _get_server_mcp_manager(conn)
    if not manager:
        raise RuntimeError("server mcp manager is not ready")
    return await manager.execute_tool(tool_name, arguments or {})


def _is_affirmative_short_reply(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return False

    negative_tokens = (
        "不可以",
        "不要",
        "别拍",
        "不拍",
        "先别",
        "不能拍",
        "不让拍",
        "别现在拍",
    )
    if _contains_any(norm, negative_tokens):
        return False

    if norm in {
        "好",
        "好的",
        "好啊",
        "好呀",
        "可以",
        "可以的",
        "可以拍",
        "拍吧",
        "拍",
        "开始拍",
        "行",
        "行的",
        "行啊",
        "嗯",
        "嗯嗯",
        "是",
        "对",
        "没问题",
        "同意",
        "允许",
        "准备好了",
        "我准备好了",
    }:
        return True

    if _looks_like_question_reply(norm):
        return False

    affirmative_tokens = (
        "可以拍照",
        "现在可以拍照",
        "可以拍",
        "可以拍了",
        "现在可以拍",
        "现在可以了",
        "可以了",
        "拍照吧",
        "拍一张吧",
        "拍一下吧",
        "直接拍吧",
        "没问题拍",
        "同意拍",
    )
    if _contains_any(norm, affirmative_tokens):
        return True

    affirmative_prefixes = (
        "好",
        "好的",
        "好啊",
        "好呀",
        "可以",
        "可以的",
        "可以啊",
        "可以呀",
        "行",
        "行的",
        "行啊",
        "行呀",
        "嗯",
        "嗯嗯",
        "对",
        "是",
        "没问题",
        "当然可以",
        "同意",
        "允许",
        "准备好了",
        "我准备好了",
    )
    return _starts_with_any(norm, affirmative_prefixes)


def _is_negative_short_reply(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if norm in {
        "不要",
        "先别",
        "别拍",
        "不拍",
        "还没准备好",
        "没准备好",
        "等等",
        "等一下",
        "暂时不要",
        "不可以",
        "取消",
    }:
        return True

    if not norm or len(norm) > 12:
        return False

    negative_tokens = (
        "不可以拍照",
        "现在不可以拍照",
        "还不可以拍照",
        "还不能拍照",
        "先别拍照",
    )
    return _contains_any(norm, negative_tokens)


def _get_last_assistant_text(conn) -> str:
    dialogue_items = getattr(getattr(conn, "dialogue", None), "dialogue", [])
    for item in reversed(dialogue_items):
        if getattr(item, "role", "") != "assistant":
            continue
        content = getattr(item, "content", "")
        if isinstance(content, str) and content.strip():
            return textUtils.normalize_spoken_text(content)
    return ""


def _assistant_is_waiting_for_photo_permission(conn) -> bool:
    last_text = _normalize_text_for_match(_get_last_assistant_text(conn))
    if not last_text:
        return False
    photo_tokens = (
        "拍照",
        "拍一张",
        "拍一下",
        "照一下",
        "照片",
    )
    if not _contains_any(last_text, photo_tokens):
        return False

    explicit_wait_tokens = (
        "得到肯定答复后再拍",
        "确认后再拍",
        "同意后再拍",
        "回复可以再拍",
    )
    if _contains_any(last_text, explicit_wait_tokens):
        return True

    prompt_tokens = (
        "可以",
        "能",
        "要不要",
        "要不",
        "要我",
        "帮你",
        "给你",
        "让我",
        "是否",
        "确认",
        "同意",
    )
    question_tokens = (
        "吗",
        "么",
        "嘛",
        "是否",
        "可不可以",
        "能不能",
        "要不要",
    )
    return _contains_any(last_text, prompt_tokens) and _contains_any(
        last_text, question_tokens
    )


def _update_server_photo_confirmation_state(conn, filtered_text: str) -> None:
    if not _assistant_is_waiting_for_photo_permission(conn):
        return
    if _is_affirmative_short_reply(filtered_text):
        conn._server_photo_capture_granted = True
    elif _is_negative_short_reply(filtered_text):
        conn._server_photo_capture_granted = False


async def _execute_direct_photo_intent(
    conn,
    question: str,
    raw_tool_name: str,
    timeout: int,
) -> bool:
    mcp_client = getattr(conn, "mcp_client", None)
    if not mcp_client:
        speak_txt(conn, "\u8bbe\u5907\u8fd8\u6ca1\u51c6\u5907\u597d\u62cd\u7167\u3002")
        return True

    if not await mcp_client.is_ready():
        speak_txt(
            conn,
            "\u8bbe\u5907\u62cd\u7167\u529f\u80fd\u8fd8\u6ca1\u51c6\u5907\u597d\uff0c\u8bf7\u7a0d\u540e\u518d\u8bd5\u3002",
        )
        return True

    tool_name = sanitize_tool_name(raw_tool_name)
    if timeout <= 0:
        timeout = 45

    try:
        result = await call_mcp_tool(
            conn,
            mcp_client,
            tool_name,
            {"question": question},
            timeout=timeout,
            allow_unlisted=True,
            raw_tool_name=raw_tool_name,
        )
    except TimeoutError:
        speak_txt(conn, "\u62cd\u7167\u8d85\u65f6\u4e86\uff0c\u8bf7\u518d\u8bd5\u4e00\u6b21\u3002")
        return True
    except Exception as e:
        conn.logger.bind(tag=TAG).warning(f"direct photo mcp failed: {e}")
        speak_txt(conn, f"\u62cd\u7167\u5931\u8d25\uff1a{e}")
        return True

    reply = _extract_direct_photo_reply(result) or "\u62cd\u597d\u4e86\u3002"
    speak_txt(conn, reply)
    return True


async def handle_pending_direct_photo_confirmation(
    conn, original_text: str, filtered_text: str
) -> bool:
    pending = getattr(conn, "_pending_direct_photo", None)
    if not isinstance(pending, dict):
        return False

    if _is_negative_short_reply(filtered_text):
        await send_stt_message(conn, original_text)
        conn.client_abort = False
        conn.sentence_id = str(uuid.uuid4().hex)
        conn.dialogue.put(Message(role="user", content=original_text))
        conn._pending_direct_photo = None
        speak_txt(conn, "\u597d\uff0c\u90a3\u6211\u5148\u4e0d\u62cd\u3002")
        return True

    if not _is_affirmative_short_reply(filtered_text):
        return False

    await send_stt_message(conn, original_text)
    conn.client_abort = False
    conn.sentence_id = str(uuid.uuid4().hex)
    conn.dialogue.put(Message(role="user", content=original_text))
    conn._pending_direct_photo = None
    return await _execute_direct_photo_intent(
        conn,
        pending.get("question", "\u63cf\u8ff0\u4e00\u4e0b\u770b\u5230\u7684\u7269\u54c1"),
        pending.get("raw_tool_name", "self.camera.take_photo"),
        int(pending.get("timeout", 45)),
    )


async def handle_direct_photo_navigation_intent(
    conn, original_text: str, filtered_text: str
) -> bool:
    nav_type = _classify_photo_nav_command(filtered_text)
    if not nav_type:
        return False

    shortcut_cfg = conn.config.get("device_mcp_shortcuts", {}) or {}
    if shortcut_cfg.get("enable_photo_navigation_direct", True) is False:
        return False

    await send_stt_message(conn, original_text)
    conn.client_abort = False
    conn.sentence_id = str(uuid.uuid4().hex)
    conn.dialogue.put(Message(role="user", content=original_text))

    safe_device_id = str(getattr(conn, "device_id", "") or "").strip()
    if not safe_device_id:
        speak_txt(conn, "\u8bbe\u5907\u8fde\u63a5\u4fe1\u606f\u7f3a\u5931\uff0c\u8bf7\u7a0d\u540e\u518d\u8bd5\u3002")
        return True

    if _get_server_mcp_manager(conn) is None:
        speak_txt(conn, "\u56fe\u7247\u9884\u89c8\u529f\u80fd\u8fd8\u6ca1\u51c6\u5907\u597d\uff0c\u8bf7\u7a0d\u540e\u518d\u8bd5\u3002")
        return True

    try:
        if nav_type == "previous":
            tool_name = str(
                shortcut_cfg.get(
                    "preview_previous_tool_name",
                    "xiaozhi_preview_previous_photo",
                )
            ).strip() or "xiaozhi_preview_previous_photo"
            result = await _execute_server_mcp_tool_direct(
                conn,
                tool_name,
                {"device_id": safe_device_id},
            )
            default_reply = "\u5df2\u7ecf\u5207\u5230\u4e0a\u4e00\u5f20\u4e86\u3002"
        else:
            tool_name = str(
                shortcut_cfg.get(
                    "preview_latest_tool_name",
                    "xiaozhi_preview_local_file",
                )
            ).strip() or "xiaozhi_preview_local_file"
            result = await _execute_server_mcp_tool_direct(
                conn,
                tool_name,
                {"device_id": safe_device_id, "photo_index": 0},
            )
            default_reply = "\u5df2\u7ecf\u6253\u5f00\u6700\u8fd1\u4e00\u5f20\u7167\u7247\u4e86\u3002"
    except Exception as e:
        conn.logger.bind(tag=TAG).warning(f"direct photo navigation failed: {e}")
        speak_txt(conn, f"\u6253\u5f00\u7167\u7247\u5931\u8d25\uff1a{e}")
        return True

    payload = _extract_server_mcp_payload(result)
    if isinstance(payload, dict) and payload.get("success") is False:
        msg = str(payload.get("message", "")).strip() or "\u6253\u5f00\u7167\u7247\u5931\u8d25\u3002"
        speak_txt(conn, msg)
        return True

    reply = _extract_text_from_result_payload(payload) or default_reply
    speak_txt(conn, reply)
    return True


async def handle_direct_photo_intent(conn, original_text: str, filtered_text: str) -> bool:
    if not _is_direct_photo_command(filtered_text):
        return False

    if _assistant_is_waiting_for_photo_permission(conn) and (
        _is_affirmative_short_reply(filtered_text)
        or _is_negative_short_reply(filtered_text)
    ):
        return False

    shortcut_cfg = conn.config.get("device_mcp_shortcuts", {}) or {}
    if shortcut_cfg.get("enable_photo_direct", True) is False:
        return False

    await send_stt_message(conn, original_text)
    conn.client_abort = False
    conn.sentence_id = str(uuid.uuid4().hex)
    conn.dialogue.put(Message(role="user", content=original_text))

    raw_tool_name = str(
        shortcut_cfg.get("take_photo_tool_name", "self.camera.take_photo")
    ).strip() or "self.camera.take_photo"
    timeout = int(shortcut_cfg.get("photo_timeout", 45))

    default_question = str(
        shortcut_cfg.get(
            "default_photo_question",
            "\u63cf\u8ff0\u4e00\u4e0b\u770b\u5230\u7684\u7269\u54c1",
        )
    ).strip() or "\u63cf\u8ff0\u4e00\u4e0b\u770b\u5230\u7684\u7269\u54c1"
    question = _build_direct_photo_question(original_text, default_question)
    conn._pending_direct_photo = {
        "question": question,
        "raw_tool_name": raw_tool_name,
        "timeout": timeout,
    }
    speak_txt(conn, "\u53ef\u4ee5\u62cd\u7167\u5417\uff1f")
    return True


async def process_intent_result(conn, intent_result, original_text):
    """处理意图识别结果"""
    try:
        # 尝试将结果解析为JSON
        intent_data = json.loads(intent_result)

        # 检查是否有function_call
        if "function_call" in intent_data:
            # 直接从意图识别获取了function_call
            conn.logger.bind(tag=TAG).debug(
                f"检测到function_call格式的意图结果: {intent_data['function_call']['name']}"
            )
            function_name = intent_data["function_call"]["name"]
            if function_name == "continue_chat":
                return False

            if function_name == "result_for_context":
                await send_stt_message(conn, original_text)
                conn.client_abort = False
                
                def process_context_result():
                    conn.dialogue.put(Message(role="user", content=original_text))
                    
                    from core.utils.current_time import get_current_time_info

                    current_time, today_date, today_weekday, lunar_date = get_current_time_info()
                    
                    # 构建带上下文的基础提示
                    context_prompt = f"""当前时间：{current_time}
                                        今天日期：{today_date} ({today_weekday})
                                        今天农历：{lunar_date}

                                        请根据以上信息回答用户的问题：{original_text}"""
                    
                    response = conn.intent.replyResult(context_prompt, original_text)
                    speak_txt(conn, response)
                
                conn.executor.submit(process_context_result)
                return True

            function_args = {}
            if "arguments" in intent_data["function_call"]:
                function_args = intent_data["function_call"]["arguments"]
                if function_args is None:
                    function_args = {}
            # 确保参数是字符串格式的JSON
            if isinstance(function_args, dict):
                function_args = json.dumps(function_args)

            function_call_data = {
                "name": function_name,
                "id": str(uuid.uuid4().hex),
                "arguments": function_args,
            }

            await send_stt_message(conn, original_text)
            conn.client_abort = False

            # 使用executor执行函数调用和结果处理
            def process_function_call():
                conn.dialogue.put(Message(role="user", content=original_text))

                # 使用统一工具处理器处理所有工具调用
                try:
                    result = asyncio.run_coroutine_threadsafe(
                        conn.func_handler.handle_llm_function_call(
                            conn, function_call_data
                        ),
                        conn.loop,
                    ).result()
                except Exception as e:
                    conn.logger.bind(tag=TAG).error(f"工具调用失败: {e}")
                    result = ActionResponse(
                        action=Action.ERROR, result=str(e), response=str(e)
                    )

                if result:
                    if result.action == Action.RESPONSE:  # 直接回复前端
                        text = result.response
                        if text is not None:
                            speak_txt(conn, text)
                    elif result.action == Action.REQLLM:  # 调用函数后再请求llm生成回复
                        text = result.result
                        conn.dialogue.put(Message(role="tool", content=text))
                        llm_result = conn.intent.replyResult(text, original_text)
                        if llm_result is None:
                            llm_result = text
                        speak_txt(conn, llm_result)
                    elif (
                        result.action == Action.NOTFOUND
                        or result.action == Action.ERROR
                    ):
                        text = result.result
                        if text is not None:
                            speak_txt(conn, text)
                    elif function_name != "play_music":
                        # For backward compatibility with original code
                        # 获取当前最新的文本索引
                        text = result.response
                        if text is None:
                            text = result.result
                        if text is not None:
                            speak_txt(conn, text)

            # 将函数执行放在线程池中
            conn.executor.submit(process_function_call)
            return True
        return False
    except json.JSONDecodeError as e:
        conn.logger.bind(tag=TAG).error(f"处理意图结果时出错: {e}")
        return False


def speak_txt(conn, text):
    text = textUtils.normalize_spoken_text(text)
    if not text:
        return

    # 记录文本
    conn.tts_MessageText = text

    conn.tts.tts_text_queue.put(
        TTSMessageDTO(
            sentence_id=conn.sentence_id,
            sentence_type=SentenceType.FIRST,
            content_type=ContentType.ACTION,
        )
    )
    conn.tts.tts_one_sentence(conn, ContentType.TEXT, content_detail=text)
    conn.tts.tts_text_queue.put(
        TTSMessageDTO(
            sentence_id=conn.sentence_id,
            sentence_type=SentenceType.LAST,
            content_type=ContentType.ACTION,
        )
    )
    conn.dialogue.put(Message(role="assistant", content=text))
