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
        self.api_url = config.get("api_url", "https://api.openai.com/v1/audio/speech")
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

        response = requests.post(
            self.api_url, json=data, headers=headers, timeout=self.timeout
        )
        response_body = response.content or b""
        content_type = (response.headers.get("content-type") or "").lower()

        if response.status_code != 200:
            raise Exception(
                f"OpenAI TTS request failed: {response.status_code} - "
                f"{self._response_error_text(response)}"
            )

        if self._is_json_response(content_type, response_body):
            raise Exception(
                "OpenAI TTS returned JSON instead of audio: "
                f"{self._response_error_text(response)}"
            )

        if not response_body:
            raise Exception("OpenAI TTS returned empty audio")

        if output_file:
            with open(output_file, "wb") as audio_file:
                audio_file.write(response_body)
        else:
            return response_body

    @staticmethod
    def _is_json_response(content_type, response_body):
        if "application/json" in content_type:
            return True
        stripped_body = response_body.lstrip()
        return stripped_body.startswith((b"{", b"["))

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
