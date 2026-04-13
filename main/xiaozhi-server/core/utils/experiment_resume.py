import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.logger import setup_logging
from core.utils import textUtils

TAG = __name__
logger = setup_logging()

RESUME_INTENT_PATTERNS = (
    re.compile(
        r"(继续|接着|恢复)(刚才|之前|上次)?(没做完|未完成)?的?(实验|实验流程|当前实验|当前流程|步骤)"
    ),
    re.compile(r"(继续|接着)(刚才|之前|上次)?没做完"),
)

EXPORT_RECORD_PATTERNS = (
    re.compile(r"(实验结束|结束实验|结束当前实验)"),
    re.compile(r"(生成|导出|写出|保存)(实验记录|记录yaml|记录YAML|yaml|YAML)"),
)

TURN_SPLIT_RE = re.compile(r"(?=^\[[^\]]+\] \[TURN_START\])", re.MULTILINE)
TURN_END_RE = re.compile(r"^\[[^\]]+\] \[TURN_END\] chars=.*$", re.MULTILINE)
USER_LINE_RE = re.compile(r"^\[[^\]]+\] \[USER\] (?P<user>.*)$", re.MULTILINE)


class _PathFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _safe_filename(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    return text[:120] if len(text) > 120 else text


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def is_resume_experiment_request(text: Any) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in RESUME_INTENT_PATTERNS)


def is_experiment_record_request(text: Any) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in EXPORT_RECORD_PATTERNS)


def should_load_device_log_context(text: Any) -> bool:
    return is_resume_experiment_request(text) or is_experiment_record_request(text)


def _iter_codex_llm_configs(config: Dict[str, Any]):
    llm_map = config.get("LLM", {}) or {}
    if not isinstance(llm_map, dict):
        return

    preferred = str(config.get("codex_app", {}).get("llm_name", "")).strip()
    seen = set()

    if preferred:
        preferred_cfg = llm_map.get(preferred)
        if (
            isinstance(preferred_cfg, dict)
            and str(preferred_cfg.get("type", "")).strip() == "codex"
        ):
            seen.add(preferred)
            yield preferred, preferred_cfg

    for name, llm_cfg in llm_map.items():
        if name in seen:
            continue
        if isinstance(llm_cfg, dict) and str(llm_cfg.get("type", "")).strip() == "codex":
            yield str(name), llm_cfg


def _format_template_path(template: str, replacements: Dict[str, str]) -> str:
    text = str(template or "").strip()
    if not text:
        return ""
    try:
        return text.format_map(_PathFormatDict(replacements))
    except Exception:
        return text


def resolve_experiment_log_paths(config: Dict[str, Any], device_id: str) -> List[Path]:
    normalized_device_id = str(device_id or "").strip()
    if not normalized_device_id:
        return []

    safe_device_id = _safe_filename(normalized_device_id)
    paths: List[Path] = []
    seen = set()

    for _, llm_cfg in _iter_codex_llm_configs(config):
        template = str(llm_cfg.get("stream_log_path", "")).strip()
        if not template:
            continue
        for candidate_device_id in (safe_device_id, normalized_device_id):
            path_text = _format_template_path(
                template,
                {
                    "device_id": candidate_device_id,
                    "session_key": safe_device_id or "resume",
                },
            )
            if not path_text:
                continue
            try:
                path = Path(path_text)
            except Exception:
                continue
            key = str(path).lower()
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)

    return paths


def _select_existing_log_path(paths: List[Path]) -> Optional[Path]:
    existing = [path for path in paths if path.exists() and path.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda item: item.stat().st_mtime)


def _parse_turns(log_text: str) -> List[Dict[str, str]]:
    turns: List[Dict[str, str]] = []
    for chunk in TURN_SPLIT_RE.split(log_text or ""):
        if "[TURN_START]" not in chunk:
            continue
        end_match = TURN_END_RE.search(chunk)
        if not end_match:
            continue
        body = chunk[: end_match.start()]
        user_match = USER_LINE_RE.search(body)
        if not user_match:
            continue

        user_text = (user_match.group("user") or "").strip()
        assistant_text = body[user_match.end() :].strip()
        if assistant_text:
            assistant_text = assistant_text.lstrip("\r\n").strip()

        if not user_text and not assistant_text:
            continue
        turns.append(
            {
                "user": user_text,
                "assistant": assistant_text,
            }
        )
    return turns


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_assistant_text(value: str) -> str:
    compact = _compact_text(value)
    if not compact:
        return ""
    cleaned = textUtils.filter_spoken_backstage_text(compact).strip()
    if cleaned:
        return cleaned
    return compact


def _format_turn_block(turns: List[Dict[str, str]]) -> str:
    lines = [
        "恢复上下文（来自当前设备日志，可信）：",
        "这是同一设备在服务重启后的继续未完成实验请求。",
        "不要重新开始实验，也不要要求重新注册声纹；直接根据最近进度继续当前未完成步骤。",
        "最近实验记录摘录：",
    ]

    for turn in turns:
        user_text = _compact_text(turn.get("user", ""))
        assistant_text = _clean_assistant_text(turn.get("assistant", ""))
        if user_text:
            lines.append(f"学生：{user_text}")
        if assistant_text:
            lines.append(f"上一轮指导：{assistant_text}")

    return "\n".join(lines).strip()


def build_resume_context(
    config: Dict[str, Any],
    device_id: str,
    *,
    max_turns: int = 4,
    max_chars: int = 2800,
) -> Optional[Dict[str, str]]:
    candidates = resolve_experiment_log_paths(config, device_id)
    log_path = _select_existing_log_path(candidates)
    if log_path is None:
        logger.bind(tag=TAG).info(
            f"resume context skipped: no device log found for device_id={device_id}"
        )
        return None

    try:
        log_text = log_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.bind(tag=TAG).warning(f"resume log read failed: {log_path} ({exc})")
        return None

    turns = _parse_turns(log_text)
    if not turns:
        tail = _compact_text(log_text[-max_chars:])
        if not tail:
            return None
        context_text = (
            "恢复上下文（来自当前设备日志，可信）：\n"
            "这是同一设备在服务重启后的继续未完成实验请求。\n"
            "最近日志尾部：\n"
            f"{tail}"
        )
        return {
            "log_path": str(log_path),
            "context_text": context_text[:max_chars],
            "turn_count": "0",
        }

    selected = turns[-max(1, int(max_turns)) :]
    while selected:
        context_text = _format_turn_block(selected)
        if len(context_text) <= max_chars or len(selected) == 1:
            return {
                "log_path": str(log_path),
                "context_text": context_text[:max_chars],
                "turn_count": str(len(selected)),
            }
        selected = selected[1:]

    return None


def build_resume_tool_message(
    config: Dict[str, Any],
    device_id: str,
    query: str,
) -> Optional[Dict[str, str]]:
    if not should_load_device_log_context(query):
        return None

    is_record_request = is_experiment_record_request(query)
    resume_context = build_resume_context(
        config,
        device_id,
        max_turns=8 if is_record_request else 4,
        max_chars=4200 if is_record_request else 2800,
    )
    if not resume_context:
        return None

    content_lines = [resume_context["context_text"]]
    content_lines.append(f"当前设备日志文件：{resume_context['log_path']}")
    if is_record_request:
        content_lines.append(
            "当前请求涉及实验记录导出；如果当前会话缺少前面断掉步骤的数据，优先继续读取这个设备日志文件的更早内容后再补记录。"
        )
    content_lines.append("当前用户消息：" + _compact_text(query))
    content = "\n".join(content_lines).strip()
    return {
        "role": "tool",
        "tool_call_id": "resume_context",
        "content": content,
    }


def dump_resume_context(config: Dict[str, Any], device_id: str) -> str:
    context = build_resume_context(config, device_id)
    if not context:
        return "{}"
    return json.dumps(context, ensure_ascii=False, indent=2)
