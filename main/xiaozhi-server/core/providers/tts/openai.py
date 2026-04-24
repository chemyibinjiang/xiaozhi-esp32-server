import os
import queue
import asyncio
import traceback

import aiohttp
import requests

from core.utils.util import check_model_key, audio_bytes_to_data_stream, pcm_to_data_stream
from core.utils.tts import MarkdownCleaner
from core.providers.tts.base import TTSProviderBase
from core.utils import opus_encoder_utils, textUtils
from core.providers.tts.dto.dto import SentenceType, ContentType, InterfaceType
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


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

        self.stream = _as_bool(config.get("stream", False))
        self.response_format = config.get("format", "wav")
        self.audio_file_type = "pcm" if self.stream else self.response_format

        speed = config.get("speed", "1.0")
        self.speed = float(speed) if speed else 1.0
        self.output_file = config.get("output_dir", "tmp/")
        self.timeout = int(config.get("timeout", 20) or 20)

        if self.stream:
            self.interface_type = InterfaceType.SINGLE_STREAM
            self.sample_rate = int(config.get("sample_rate", 24000) or 24000)
            self.opus_encoder = opus_encoder_utils.OpusEncoderUtils(
                sample_rate=self.sample_rate,
                channels=1,
                frame_size_ms=60,
            )
            self.pcm_buffer = bytearray()
            self.before_stop_play_files = []

        model_key_msg = check_model_key("TTS", self.api_key)
        if model_key_msg:
            logger.bind(tag=TAG).error(model_key_msg)

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, text, response_format=None):
        return {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": response_format or self.response_format,
            "speed": self.speed,
        }

    def tts_text_priority_thread(self):
        if not self.stream:
            return super().tts_text_priority_thread()

        while not self.conn.stop_event.is_set():
            try:
                message = self.tts_text_queue.get(timeout=1)
                if message.sentence_type == SentenceType.FIRST:
                    self.tts_stop_request = False
                    self.processed_chars = 0
                    self.tts_text_buff = []
                    self.before_stop_play_files.clear()
                elif ContentType.TEXT == message.content_type:
                    self.tts_text_buff.append(message.content_detail)
                    segment_text = self._get_segment_text()
                    if segment_text:
                        self.to_tts_single_stream(segment_text)
                elif ContentType.FILE == message.content_type:
                    if message.content_file and os.path.exists(message.content_file):
                        self._process_audio_file_stream(
                            message.content_file,
                            callback=lambda audio_data: self.handle_audio_file(
                                audio_data, message.content_detail
                            ),
                        )

                if message.sentence_type == SentenceType.LAST:
                    self._process_remaining_text_stream(True)

            except queue.Empty:
                continue
            except Exception as e:
                logger.bind(tag=TAG).error(
                    "processing stream TTS text failed: "
                    f"{e}, type={type(e).__name__}, stack={traceback.format_exc()}"
                )

    def _process_remaining_text_stream(self, is_last=False):
        full_text = "".join(self.tts_text_buff)
        remaining_text = full_text[self.processed_chars :]
        if remaining_text:
            segment_text = textUtils.get_string_no_punctuation_or_emoji(remaining_text)
            if segment_text:
                self.to_tts_single_stream(segment_text, is_last)
                self.processed_chars = len(full_text)
            else:
                self._process_before_stop_play_files()
        else:
            self._process_before_stop_play_files()

    def to_tts_single_stream(self, text, is_last=False):
        text = MarkdownCleaner.clean_markdown(text)
        text = textUtils.filter_spoken_backstage_text(text)
        if not text:
            if is_last:
                self._process_before_stop_play_files()
            return None

        retries = 5
        while retries > 0:
            try:
                asyncio.run(self._stream_text_to_speak(text, is_last))
                return None
            except Exception as e:
                retries -= 1
                logger.bind(tag=TAG).warning(
                    f"stream TTS failed, remaining_retries={retries}, text={text}, err={e}"
                )

        logger.bind(tag=TAG).error(f"stream TTS exhausted retries: {text}")
        if is_last:
            self.tts_audio_queue.put((SentenceType.LAST, [], None))
        return None

    async def text_to_speak(self, text, output_file):
        if self.stream and isinstance(output_file, bool):
            await self._stream_text_to_speak(text, output_file)
            return None
        return await self._non_stream_text_to_speak(text, output_file)

    async def _non_stream_text_to_speak(self, text, output_file):
        response = requests.post(
            self.api_url,
            json=self._payload(text),
            headers=self._headers(),
            timeout=self.timeout,
        )
        if response.status_code == 200:
            if output_file:
                with open(output_file, "wb") as audio_file:
                    audio_file.write(response.content)
            else:
                return response.content
        else:
            raise Exception(
                f"OpenAI TTS request failed: {response.status_code} - {response.text}"
            )

    async def _stream_text_to_speak(self, text, is_last):
        frame_bytes = int(
            self.opus_encoder.sample_rate
            * self.opus_encoder.channels
            * self.opus_encoder.frame_size_ms
            / 1000
            * 2
        )

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.api_url,
                json=self._payload(text, response_format="pcm"),
                headers=self._headers(),
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(
                        f"OpenAI stream TTS request failed: {resp.status} - {await resp.text()}"
                    )

                content_type = (resp.headers.get("content-type") or "").lower()
                if "application/json" in content_type:
                    raise RuntimeError(
                        f"stream endpoint returned json instead of audio: {await resp.text()}"
                    )

                self.pcm_buffer.clear()
                self.tts_audio_queue.put((SentenceType.FIRST, [], text))
                saw_audio = False
                prefix = bytearray()

                async for chunk in resp.content.iter_any():
                    data = chunk[0] if isinstance(chunk, (list, tuple)) else chunk
                    if not data:
                        continue

                    if len(prefix) < 4:
                        need = 4 - len(prefix)
                        prefix.extend(data[:need])
                    if prefix.startswith(b"RIFF"):
                        raise RuntimeError(
                            "stream mode requested pcm, but endpoint returned wav"
                        )

                    saw_audio = True
                    self.pcm_buffer.extend(data)

                    while len(self.pcm_buffer) >= frame_bytes:
                        frame = bytes(self.pcm_buffer[:frame_bytes])
                        del self.pcm_buffer[:frame_bytes]
                        self.opus_encoder.encode_pcm_to_opus_stream(
                            frame,
                            end_of_stream=False,
                            callback=self.handle_opus,
                        )

                if not saw_audio:
                    raise RuntimeError("stream endpoint returned no audio bytes")

                if self.pcm_buffer:
                    self.opus_encoder.encode_pcm_to_opus_stream(
                        bytes(self.pcm_buffer),
                        end_of_stream=True,
                        callback=self.handle_opus,
                    )
                    self.pcm_buffer.clear()

                if is_last:
                    self._process_before_stop_play_files()

    def to_tts(self, text: str) -> list:
        text = MarkdownCleaner.clean_markdown(text)
        text = textUtils.filter_spoken_backstage_text(text)
        if not text:
            return []

        response = requests.post(
            self.api_url,
            json=self._payload(text),
            headers=self._headers(),
            timeout=self.timeout,
        )
        if response.status_code != 200:
            logger.bind(tag=TAG).error(
                f"OpenAI TTS request failed: {response.status_code}, {response.text}"
            )
            return []

        audio_datas = []
        if self.response_format == "pcm":
            pcm_to_data_stream(
                response.content,
                is_opus=True,
                callback=lambda data: audio_datas.append(data),
            )
        else:
            audio_bytes_to_data_stream(
                response.content,
                file_type=self.response_format,
                is_opus=True,
                callback=lambda data: audio_datas.append(data),
            )
        return audio_datas

    async def close(self):
        await super().close()
        if hasattr(self, "opus_encoder"):
            self.opus_encoder.close()
