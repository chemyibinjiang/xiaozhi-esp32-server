from __future__ import annotations

import json
from typing import Any


def to_plain_data(payload):
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


def try_parse_json_text(text: str):
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


def extract_server_mcp_payload(raw_result):
    data = to_plain_data(raw_result)
    if isinstance(data, str):
        parsed = try_parse_json_text(data)
        return parsed if parsed is not None else data

    if isinstance(data, dict):
        content = data.get("content")
        if isinstance(content, list):
            for item in content:
                item_data = to_plain_data(item)
                if isinstance(item_data, dict):
                    text = item_data.get("text")
                    if isinstance(text, str):
                        parsed_text = try_parse_json_text(text)
                        if parsed_text is not None:
                            return parsed_text
                        if text.strip():
                            return text.strip()
        return data

    if isinstance(data, list):
        for item in data:
            extracted = extract_server_mcp_payload(item)
            if extracted is not None:
                return extracted
    return data


def serialize_result_for_llm(payload) -> str:
    data = extract_server_mcp_payload(payload)
    if isinstance(data, (dict, list)):
        return json.dumps(data, ensure_ascii=False)
    if data is None:
        return ""
    return str(data)


def pick_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""
