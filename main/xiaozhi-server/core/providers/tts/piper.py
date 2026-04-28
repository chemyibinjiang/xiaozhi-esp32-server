import builtins
import contextlib
import io
import os
import re
import tarfile
import threading
import wave
from pathlib import Path

import requests

from config.logger import setup_logging
from core.providers.tts.base import TTSProviderBase

try:
    from piper import PiperVoice, SynthesisConfig
except ImportError:
    PiperVoice = None
    SynthesisConfig = None


TAG = __name__
logger = setup_logging()
VOICE_PATTERN = re.compile(
    r"^(?P<lang_family>[^-]+)_(?P<lang_region>[^-]+)-(?P<voice_name>[^-]+)-(?P<voice_quality>.+)$"
)
DEFAULT_DOWNLOAD_BASE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main"
)
DEFAULT_G2PW_URL = (
    "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/"
    "zh/zh_CN/_resources/g2pw.tar.gz?download=true"
)
MIRROR_G2PW_URL = (
    "https://hf-mirror.com/datasets/rhasspy/piper-checkpoints/resolve/main/"
    "zh/zh_CN/_resources/g2pw.tar.gz?download=true"
)


class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)

        self.output_file = config.get("output_dir", "tmp/")
        self.audio_file_type = "wav"
        self.format = "wav"

        requested_format = str(config.get("format", "wav") or "wav").lower()
        if requested_format != "wav":
            logger.bind(tag=TAG).warning(
                f"PiperTTS only outputs wav, overriding requested format '{requested_format}'"
            )

        self.voice_name = str(config.get("voice") or "").strip()
        self.model_dir = Path(config.get("model_dir", os.path.join("models", "piper")))
        self.model_path = self._resolve_optional_path(config.get("model_path"))
        self.config_path = self._resolve_optional_path(config.get("config_path"))
        self.auto_download = self._as_bool(config.get("auto_download", True))
        self.use_cuda = self._as_bool(config.get("use_cuda", False))
        self.download_base_url = str(
            config.get("download_base_url", DEFAULT_DOWNLOAD_BASE_URL)
            or DEFAULT_DOWNLOAD_BASE_URL
        ).rstrip("/")
        self.g2pw_url = str(
            config.get(
                "g2pw_url",
                MIRROR_G2PW_URL
                if "hf-mirror.com" in self.download_base_url
                else DEFAULT_G2PW_URL,
            )
            or DEFAULT_G2PW_URL
        ).strip()
        self.download_timeout = max(
            5.0, float(config.get("download_timeout", 120) or 120)
        )
        self.download_retries = max(1, int(config.get("download_retries", 3) or 3))
        self.normalize_audio = self._as_bool(config.get("normalize_audio", True))
        self.sentence_silence = max(
            0.0,
            self._optional_float(
                config.get("sentence_silence", config.get("sentence_silence_seconds"))
            )
            or 0.0,
        )
        self.volume = float(config.get("volume", 1.0) or 1.0)
        self.speaker_id = self._optional_int(
            config.get("speaker_id", config.get("speaker"))
        )
        self.length_scale = self._optional_float(config.get("length_scale"))
        self.noise_scale = self._optional_float(config.get("noise_scale"))
        self.noise_w_scale = self._optional_float(
            config.get("noise_w_scale", config.get("noise_w"))
        )

        self._voice_lock = threading.Lock()
        self._loaded_voice = None
        self._loaded_voice_key = None

        if PiperVoice is None or SynthesisConfig is None:
            raise ImportError(
                "PiperTTS requires the 'piper-tts' package. "
                "Install it with: python -m pip install piper-tts"
            )

    async def text_to_speak(self, text, output_file):
        voice = self._ensure_voice()
        synth_config = SynthesisConfig(
            speaker_id=self.speaker_id,
            length_scale=self.length_scale,
            noise_scale=self.noise_scale,
            noise_w_scale=self.noise_w_scale,
            normalize_audio=self.normalize_audio,
            volume=self.volume,
        )

        with self._voice_lock:
            with self._patch_g2pw_open():
                chunks = list(voice.synthesize(text, syn_config=synth_config))

        if not chunks:
            raise Exception("PiperTTS produced no audio")

        wav_bytes = self._chunks_to_wav_bytes(chunks)
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "wb") as wav_file:
                wav_file.write(wav_bytes)
            return None

        return wav_bytes

    def _ensure_voice(self):
        model_path, config_path = self._resolve_voice_paths()
        voice_key = (
            str(model_path.resolve()),
            str(config_path.resolve()),
            bool(self.use_cuda),
        )
        if self._loaded_voice is not None and self._loaded_voice_key == voice_key:
            return self._loaded_voice

        with self._voice_lock:
            if self._loaded_voice is not None and self._loaded_voice_key == voice_key:
                return self._loaded_voice

            logger.bind(tag=TAG).info(
                f"Loading Piper voice: model={model_path}, use_cuda={self.use_cuda}"
            )
            self._loaded_voice = PiperVoice.load(
                model_path=model_path,
                config_path=config_path,
                use_cuda=self.use_cuda,
                download_dir=self.model_dir,
            )
            self._ensure_voice_resources(self._loaded_voice)
            self._loaded_voice_key = voice_key
            return self._loaded_voice

    def _resolve_voice_paths(self):
        if self.model_path is not None:
            model_path = self.model_path
            config_path = self.config_path or Path(f"{model_path}.json")
        else:
            if not self.voice_name:
                raise ValueError(
                    "PiperTTS requires either 'voice' or 'model_path' in config"
                )

            self.model_dir.mkdir(parents=True, exist_ok=True)
            model_path = self.model_dir / f"{self.voice_name}.onnx"
            config_path = self.config_path or self.model_dir / f"{self.voice_name}.onnx.json"

            if self.auto_download and (not model_path.exists() or not config_path.exists()):
                logger.bind(tag=TAG).info(
                    f"Piper voice missing locally, downloading '{self.voice_name}' to {self.model_dir}"
                )
                self._download_voice_files(self.voice_name, model_path, config_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Piper model not found: {model_path}. "
                "Set 'voice' with auto_download=true, or provide 'model_path'."
            )

        if not config_path.exists():
            raise FileNotFoundError(
                f"Piper config not found: {config_path}. "
                "Set 'voice' with auto_download=true, or provide 'config_path'."
            )

        return model_path, config_path

    def _chunks_to_wav_bytes(self, chunks):
        first_chunk = chunks[0]
        sample_rate = first_chunk.sample_rate
        sample_width = first_chunk.sample_width
        sample_channels = first_chunk.sample_channels
        silence_bytes = b""

        if self.sentence_silence > 0:
            silence_frames = int(sample_rate * self.sentence_silence)
            silence_bytes = b"\x00" * silence_frames * sample_width * sample_channels

        pcm_parts = []
        for index, chunk in enumerate(chunks):
            pcm_parts.append(chunk.audio_int16_bytes)
            if silence_bytes and index < len(chunks) - 1:
                pcm_parts.append(silence_bytes)

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(sample_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"".join(pcm_parts))

        return wav_io.getvalue()

    def _ensure_voice_resources(self, voice):
        phoneme_type_name = getattr(getattr(voice, "config", None), "phoneme_type", None)
        phoneme_type_name = getattr(phoneme_type_name, "name", str(phoneme_type_name))
        if phoneme_type_name != "PINYIN":
            return

        g2pw_dir = self.model_dir / "g2pW"
        g2pw_model_path = g2pw_dir / "g2pw.onnx"
        if g2pw_model_path.exists():
            return

        logger.bind(tag=TAG).info(
            f"Piper Chinese resources missing locally, downloading g2pW to {g2pw_dir}"
        )
        archive_path = self.model_dir / "g2pw.tar.gz"
        self._download_file(self.g2pw_url, archive_path)
        g2pw_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar_file:
            tar_file.extractall(path=g2pw_dir)
        archive_path.unlink(missing_ok=True)

    def _download_voice_files(self, voice_name, model_path, config_path):
        voice_match = VOICE_PATTERN.match(voice_name)
        if not voice_match:
            raise ValueError(
                "Invalid Piper voice name. Expected format like 'zh_CN-xiao_ya-medium'."
            )

        lang_family = voice_match.group("lang_family")
        lang_code = f"{lang_family}_{voice_match.group('lang_region')}"
        voice_name_only = voice_match.group("voice_name")
        voice_quality = voice_match.group("voice_quality")
        base_url = (
            f"{self.download_base_url}/{lang_family}/{lang_code}/"
            f"{voice_name_only}/{voice_quality}/{voice_name}"
        )

        self._download_file(f"{base_url}.onnx?download=true", model_path)
        self._download_file(f"{base_url}.onnx.json?download=true", config_path)

    def _download_file(self, url, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_name(destination.name + ".part")
        last_error = None

        for attempt in range(1, self.download_retries + 1):
            try:
                with requests.get(
                    url,
                    stream=True,
                    timeout=(20, self.download_timeout),
                ) as response:
                    response.raise_for_status()
                    with open(temp_path, "wb") as output_file:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                output_file.write(chunk)

                temp_path.replace(destination)
                return
            except requests.RequestException as exc:
                last_error = exc
                if temp_path.exists():
                    temp_path.unlink()
                if attempt >= self.download_retries:
                    break
                logger.bind(tag=TAG).warning(
                    f"Piper download retry {attempt}/{self.download_retries} failed for {destination.name}: {exc}"
                )

        raise last_error or Exception(f"Failed to download Piper file from {url}")

    @contextlib.contextmanager
    def _patch_g2pw_open(self):
        original_open = builtins.open

        def patched_open(file, mode="r", *args, **kwargs):
            if (
                "b" not in mode
                and kwargs.get("encoding") is None
                and isinstance(file, (str, os.PathLike))
            ):
                try:
                    file_path = str(Path(file).resolve()).lower()
                except OSError:
                    file_path = str(file).lower()
                if "g2pw" in file_path:
                    kwargs["encoding"] = "utf-8"
            return original_open(file, mode, *args, **kwargs)

        builtins.open = patched_open
        try:
            yield
        finally:
            builtins.open = original_open

    @staticmethod
    def _resolve_optional_path(path_value):
        path_text = str(path_value or "").strip()
        if not path_text:
            return None
        return Path(path_text)

    @staticmethod
    def _as_bool(value):
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _optional_int(value):
        if value is None or str(value).strip() == "":
            return None
        return int(value)

    @staticmethod
    def _optional_float(value):
        if value is None or str(value).strip() == "":
            return None
        return float(value)
