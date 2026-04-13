import json
from typing import Dict, Iterable

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
        self.stream = bool(self.config.get("stream", True))

        if not self.endpoint:
            raise ValueError("codex_remote endpoint is required")

    def _post(self, session_id, dialogue, kwargs, stream=False):
        payload = {
            "session_id": str(session_id or ""),
            "dialogue": dialogue or [],
            "kwargs": kwargs or {},
        }
        if stream:
            payload["stream"] = True

        response = requests.post(
            self.endpoint,
            json=payload,
            timeout=self.timeout,
            stream=stream,
        )
        response.raise_for_status()
        return response

    def _iter_stream(self, response) -> Iterable[object]:
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue

            item = json.loads(raw_line)
            kind = str(item.get("kind", "")).strip().lower()
            if kind == "event":
                yield item.get("data")
                continue
            if kind == "text":
                text = item.get("data", "")
                if text is None:
                    text = ""
                yield str(text)
                continue
            if kind == "error":
                raise RuntimeError(str(item.get("message", "codex remote stream failed")))
            if kind == "done":
                return

    def _response_json(self, session_id, dialogue, kwargs):
        response = self._post(session_id, dialogue, kwargs, stream=False)
        try:
            return response.json()
        finally:
            response.close()

    def response(self, session_id, dialogue, **kwargs):
        if self.stream:
            response = None
            yielded_any = False
            streamed_text_parts = []
            try:
                response = self._post(session_id, dialogue, kwargs, stream=True)
                for token in self._iter_stream(response):
                    if isinstance(token, dict):
                        if self.include_events:
                            yielded_any = True
                            yield token
                    else:
                        yielded_any = True
                        text_token = str(token)
                        streamed_text_parts.append(text_token)
                        yield text_token
                return
            except Exception as exc:
                if yielded_any:
                    logger.bind(tag=TAG).warning(
                        f"codex_remote streaming interrupted after partial output, trying JSON fallback: {exc}"
                    )
                else:
                    logger.bind(tag=TAG).warning(
                        f"codex_remote streaming unavailable, fallback to JSON: {exc}"
                    )

                # Try one-shot fallback even after partial stream. If full text starts
                # with already-streamed prefix, only emit the remaining suffix.
                try:
                    result = self._response_json(session_id, dialogue, kwargs)
                    events = result.get("events", [])
                    if self.include_events and isinstance(events, list):
                        for item in events:
                            if isinstance(item, dict):
                                yield item

                    full_text = result.get("text", "")
                    if full_text is None:
                        full_text = ""
                    if not isinstance(full_text, str):
                        full_text = json.dumps(full_text, ensure_ascii=False)

                    streamed_prefix = "".join(streamed_text_parts)
                    if full_text and streamed_prefix and full_text.startswith(streamed_prefix):
                        remainder = full_text[len(streamed_prefix) :]
                        if remainder:
                            yield remainder
                    elif full_text and not streamed_prefix:
                        yield full_text
                    return
                except Exception as fallback_exc:
                    logger.bind(tag=TAG).error(
                        f"codex_remote JSON fallback failed after stream error: {fallback_exc}"
                    )
                    return
            finally:
                if response is not None:
                    response.close()

        try:
            result = self._response_json(session_id, dialogue, kwargs)
        except Exception as exc:
            logger.bind(tag=TAG).error(f"codex_remote response failed: {exc}")
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
