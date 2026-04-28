import asyncio
import json

import requests

from core.utils.util import check_model_key
from core.providers.tts.base import TTSProviderBase
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)
        self.api_key = config.get("api_key")
        self.api_url = self._resolve_api_url(
            config.get("api_url", "https://api.openai.com/v1/audio/speech")
        )
        self.model = config.get("model", "tts-1")
        if config.get("private_voice"):
            self.voice = config.get("private_voice")
        else:
            self.voice = config.get("voice", "alloy")
        self.response_format = config.get("response_format") or config.get(
            "format", "wav"
        )
        self.lang_code = config.get("lang_code")
        self.audio_file_type = self.response_format
        self.sample_rate = config.get("sample_rate")
        self.timeout = float(config.get("timeout", 30))
        self.max_retries = max(1, int(config.get("max_retries", 3) or 3))
        self.retry_backoff_seconds = float(
            config.get("retry_backoff_seconds", 0.8) or 0.8
        )

        speed = config.get("speed", "1.0")
        self.speed = float(speed) if speed else 1.0

        self.output_file = config.get("output_dir", "tmp/")
        model_key_msg = check_model_key("TTS", self.api_key)
        if model_key_msg:
            logger.bind(tag=TAG).error(model_key_msg)

    async def text_to_speak(self, text, output_file):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": self.response_format,
            "speed": self.speed,
        }
        if self.lang_code:
            data["lang_code"] = self.lang_code
        if self.sample_rate:
            data["sample_rate"] = int(self.sample_rate)

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            response = requests.post(
                self.api_url, json=data, headers=headers, timeout=self.timeout
            )
            response_body = response.content or b""
            content_type = (response.headers.get("content-type") or "").lower()

            if response.status_code == 200 and not self._is_json_response(
                content_type, response_body
            ):
                if not response_body:
                    last_error = Exception("OpenAI TTS returned empty audio")
                else:
                    if output_file:
                        with open(output_file, "wb") as audio_file:
                            audio_file.write(response_body)
                    else:
                        return response_body
                    return None
            else:
                if response.status_code != 200:
                    last_error = Exception(
                        f"OpenAI TTS request failed: {response.status_code} - "
                        f"{self._response_error_text(response)}"
                    )
                else:
                    last_error = Exception(
                        "OpenAI TTS returned JSON instead of audio: "
                        f"{self._response_error_text(response)}"
                    )

            if attempt < self.max_retries and self._is_retryable_response(
                response, response_body
            ):
                await asyncio.sleep(self.retry_backoff_seconds * attempt)
                continue
            raise last_error

        raise last_error or Exception("OpenAI TTS request failed")

    @staticmethod
    def _is_json_response(content_type, response_body):
        if "application/json" in content_type:
            return True
        stripped_body = response_body.lstrip()
        return stripped_body.startswith((b"{", b"["))

    @staticmethod
    def _resolve_api_url(api_url):
        url = str(api_url or "").strip()
        if not url:
            return "https://api.openai.com/v1/audio/speech"
        lowered = url.lower().rstrip("/")
        if lowered.endswith("/audio/speech"):
            return url.rstrip("/")
        if lowered.endswith("/v1"):
            return url.rstrip("/") + "/audio/speech"
        return url

    @classmethod
    def _is_retryable_response(cls, response, response_body):
        if response.status_code in {429, 500, 502, 503, 504}:
            return True
        if not cls._is_json_response(
            (response.headers.get("content-type") or "").lower(),
            response_body,
        ):
            return False
        error_text = cls._response_error_text(response).lower()
        retryable_markers = (
            "service load is too high",
            "try again later",
            "rate limit",
            "too many requests",
            "temporarily unavailable",
        )
        return any(marker in error_text for marker in retryable_markers)

    @staticmethod
    def _response_error_text(response):
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError):
            return response.text

        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                code = error.get("code")
                if message and code:
                    return f"{code}: {message}"
                if message:
                    return message
            message = payload.get("message")
            if message:
                return str(message)
        return json.dumps(payload, ensure_ascii=False)
