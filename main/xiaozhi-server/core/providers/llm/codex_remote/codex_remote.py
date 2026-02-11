import json
from typing import Dict

import requests

from config.logger import setup_logging
from core.providers.llm.base import LLMProviderBase

TAG = __name__
logger = setup_logging()


class LLMProvider(LLMProviderBase):
    def __init__(self, config: Dict):
        self.config = config or {}
        self.endpoint = str(self.config.get("endpoint", "")).strip()
        self.timeout = float(self.config.get("timeout", 300))
        self.include_events = bool(self.config.get("include_events", True))

        if not self.endpoint:
            raise ValueError("codex_remote endpoint is required")

    def _do_request(self, session_id, dialogue, kwargs):
        payload = {
            "session_id": str(session_id or ""),
            "dialogue": dialogue or [],
            "kwargs": kwargs or {},
        }
        response = requests.post(
            self.endpoint,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def response(self, session_id, dialogue, **kwargs):
        try:
            result = self._do_request(session_id, dialogue, kwargs)
        except Exception as exc:
            logger.bind(tag=TAG).error(f"codex_remote response failed: {exc}")
            yield "[Codex remote response error]"
            return

        events = result.get("events", [])
        if self.include_events and isinstance(events, list):
            for item in events:
                if isinstance(item, dict):
                    yield item

        text = result.get("text", "")
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)

        if text:
            yield text

    def response_with_functions(self, session_id, dialogue, functions=None, **kwargs):
        for token in self.response(session_id, dialogue, **kwargs):
            if isinstance(token, dict):
                yield token
            else:
                yield token, None
