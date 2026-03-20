import asyncio
import time
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

import aiohttp
import requests

from config.logger import setup_logging
from core.utils.cache.config import CacheType
from core.utils.cache.manager import cache_manager

TAG = __name__
logger = setup_logging()


class VoiceprintProvider:
    """声纹识别服务提供者。

    支持两种模式：
    - 静态模式（兼容旧逻辑）：使用 speakers + identify
    - 动态模式（新增）：首次自动注册 master_speaker，后续仅对 master 进行鉴权
    """

    def __init__(self, config: dict):
        self.original_url = config.get("url", "")
        self.speakers = config.get("speakers", [])
        self.speaker_map = self._parse_speakers()
        self.similarity_threshold = float(config.get("similarity_threshold", 0.4))

        # 新增：动态模式配置
        self.dynamic_mode = self._as_bool(
            config.get("dynamic_mode", config.get("dynamic_master", False))
        )
        self.dynamic_master_speaker_id = str(
            config.get("dynamic_master_speaker_id", "master_speaker")
        )
        self.dynamic_master_name = str(
            config.get("dynamic_master_name", "主说话人")
        )
        self.dynamic_registration_required_samples = max(
            1, int(config.get("dynamic_registration_required_samples", 3))
        )
        self.dynamic_registration_reply = str(
            config.get(
                "dynamic_registration_reply",
                "你好，已记录你的声音，之后我只听你的。",
            )
        )
        self.dynamic_reject_reply = str(config.get("dynamic_reject_reply", ""))
        self.dynamic_fail_open = self._as_bool(config.get("dynamic_fail_open", False))

        # 运行态
        self._dynamic_registered = False
        self._dynamic_registered_samples = 0
        self._dynamic_lock = asyncio.Lock()

        # API
        self.base_url: Optional[str] = None
        self.api_url: Optional[str] = None
        self.identify_url: Optional[str] = None
        self.register_url: Optional[str] = None
        self.api_key: Optional[str] = None
        self.speaker_ids = []
        self.enabled = False

        if not self.original_url:
            logger.bind(tag=TAG).warning("声纹识别URL未配置，声纹识别将被禁用")
            return

        parsed_url = urlparse(self.original_url)
        self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        self.identify_url = f"{self.base_url}/voiceprint/identify"
        self.register_url = f"{self.base_url}/voiceprint/register"
        self.api_url = self.identify_url

        query_params = parse_qs(parsed_url.query or "")
        self.api_key = query_params.get("key", [""])[0]
        if not self.api_key:
            logger.bind(tag=TAG).error("URL中未找到key参数，声纹识别将被禁用")
            return

        # 静态模式：提取 speaker_ids；动态模式不依赖预配置 speakers
        if not self.dynamic_mode:
            for speaker_str in self.speakers:
                try:
                    parts = speaker_str.split(",", 2)
                    if len(parts) >= 1:
                        speaker_id = parts[0].strip()
                        if speaker_id:
                            self.speaker_ids.append(speaker_id)
                except Exception:
                    continue

            if not self.speaker_ids:
                logger.bind(tag=TAG).warning("未配置有效的说话人，声纹识别将被禁用")
                return

        if self._check_server_health():
            self.enabled = True
            if self.dynamic_mode:
                logger.bind(tag=TAG).info(
                    "声纹识别已启用（动态模式）: "
                    f"register={self.register_url}, identify={self.identify_url}, "
                    f"required_samples={self.dynamic_registration_required_samples}, "
                    f"threshold={self.similarity_threshold}"
                )
            else:
                logger.bind(tag=TAG).info(
                    "声纹识别已启用（静态模式）: "
                    f"identify={self.identify_url}, "
                    f"speakers={len(self.speaker_ids)}, "
                    f"threshold={self.similarity_threshold}"
                )
        else:
            logger.bind(tag=TAG).warning(
                f"声纹识别服务器不可用，声纹识别已禁用: {self.api_url}"
            )

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    def _parse_speakers(self) -> Dict[str, Dict[str, str]]:
        """解析说话人配置。"""
        speaker_map = {}
        for speaker_str in self.speakers:
            try:
                parts = speaker_str.split(",", 2)
                if len(parts) >= 3:
                    speaker_id, name, description = (
                        parts[0].strip(),
                        parts[1].strip(),
                        parts[2].strip(),
                    )
                    speaker_map[speaker_id] = {
                        "name": name,
                        "description": description,
                    }
            except Exception as e:
                logger.bind(tag=TAG).warning(
                    f"解析说话人配置失败: {speaker_str}, 错误: {e}"
                )
        return speaker_map

    def _check_server_health(self) -> bool:
        """检查声纹识别服务器健康状态。"""
        if not self.base_url or not self.api_key:
            return False

        cache_key = f"{self.base_url}:{self.api_key}"
        cached_result = cache_manager.get(CacheType.VOICEPRINT_HEALTH, cache_key)
        if cached_result is not None:
            logger.bind(tag=TAG).debug(f"使用缓存的健康状态: {cached_result}")
            return cached_result

        logger.bind(tag=TAG).info("执行声纹服务器健康检查")
        is_healthy = False
        try:
            health_url = f"{self.base_url}/voiceprint/health?key={self.api_key}"
            response = requests.get(health_url, timeout=3)
            if response.status_code == 200:
                result = response.json()
                is_healthy = result.get("status") == "healthy"
                if is_healthy:
                    logger.bind(tag=TAG).info("声纹识别服务器健康检查通过")
                else:
                    logger.bind(tag=TAG).warning(f"声纹服务状态异常: {result}")
            else:
                logger.bind(tag=TAG).warning(
                    f"声纹服务健康检查失败: HTTP {response.status_code}"
                )
        except requests.exceptions.ConnectTimeout:
            logger.bind(tag=TAG).warning("声纹服务连接超时")
        except requests.exceptions.ConnectionError:
            logger.bind(tag=TAG).warning("声纹服务连接被拒绝")
        except Exception as e:
            logger.bind(tag=TAG).warning(f"声纹服务健康检查异常: {e}")

        cache_manager.set(CacheType.VOICEPRINT_HEALTH, cache_key, is_healthy)
        logger.bind(tag=TAG).info(f"健康检查结果已缓存: {is_healthy}")
        return is_healthy

    async def evaluate_voiceprint(
        self, audio_data: bytes, session_id: str
    ) -> Dict[str, Any]:
        """新增：返回完整鉴权决策（供后续流程接入）。"""
        decision: Dict[str, Any] = {
            "enabled": self.enabled,
            "mode": "dynamic" if self.dynamic_mode else "static",
            "status": "disabled",
            "allow_chat": True,
            "speaker_name": None,
            "score": None,
            "need_register_prompt": False,
            "register_prompt_text": "",
            "reject_prompt_text": "",
            "reason": "",
        }

        if not self.enabled or not self.api_key:
            decision["status"] = "disabled"
            decision["reason"] = "voiceprint disabled or not configured"
            return decision

        if self.dynamic_mode:
            return await self._evaluate_dynamic(audio_data, session_id, decision)
        return await self._evaluate_static(audio_data, session_id, decision)

    async def _evaluate_dynamic(
        self, audio_data: bytes, session_id: str, decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """动态模式：首次自动注册，后续鉴权。"""
        async with self._dynamic_lock:
            # 阶段1：首次自动注册（可配置需采样次数，默认3）
            if not self._dynamic_registered:
                register_ok = await self._register_master(audio_data, session_id)
                if not register_ok:
                    decision["status"] = "register_error"
                    decision["allow_chat"] = self.dynamic_fail_open
                    decision["reason"] = "dynamic register failed"
                    return decision

                self._dynamic_registered_samples += 1
                if self._dynamic_registered_samples < self.dynamic_registration_required_samples:
                    decision["status"] = "registering"
                    decision["allow_chat"] = False
                    decision["reason"] = (
                        f"collecting samples "
                        f"{self._dynamic_registered_samples}/"
                        f"{self.dynamic_registration_required_samples}"
                    )
                    return decision

                self._dynamic_registered = True
                decision["status"] = "registered"
                decision["allow_chat"] = False
                decision["speaker_name"] = self.dynamic_master_name
                decision["need_register_prompt"] = True
                decision["register_prompt_text"] = self.dynamic_registration_reply
                decision["reason"] = "dynamic register completed"
                return decision

            # 阶段2：仅对 master_speaker 做鉴权
            identify = await self._identify_by_speaker_ids(
                audio_data, [self.dynamic_master_speaker_id]
            )
            if not identify["ok"]:
                decision["status"] = "verify_error"
                decision["allow_chat"] = self.dynamic_fail_open
                decision["reason"] = identify["reason"]
                return decision

            score = float(identify["score"] or 0.0)
            speaker_id = identify["speaker_id"]
            decision["score"] = score

            if (
                speaker_id == self.dynamic_master_speaker_id
                and score >= self.similarity_threshold
            ):
                decision["status"] = "accepted"
                decision["allow_chat"] = True
                decision["speaker_name"] = self.dynamic_master_name
                decision["reason"] = "voiceprint matched"
                return decision

            decision["status"] = "rejected"
            decision["allow_chat"] = False
            decision["speaker_name"] = "未知说话人"
            decision["reject_prompt_text"] = self.dynamic_reject_reply
            decision["reason"] = (
                f"voiceprint mismatch, speaker_id={speaker_id}, score={score:.3f}, "
                f"threshold={self.similarity_threshold}"
            )
            return decision

    async def _evaluate_static(
        self, audio_data: bytes, session_id: str, decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """静态模式（兼容现有行为）：识别说话人名称。"""
        identify = await self._identify_by_speaker_ids(audio_data, self.speaker_ids)
        if not identify["ok"]:
            decision["status"] = "identify_error"
            decision["allow_chat"] = True
            decision["reason"] = identify["reason"]
            return decision

        score = float(identify["score"] or 0.0)
        speaker_id = identify["speaker_id"]
        decision["score"] = score

        if score < self.similarity_threshold:
            decision["status"] = "unknown"
            decision["allow_chat"] = True
            decision["speaker_name"] = "未知说话人"
            decision["reason"] = (
                f"score below threshold: {score:.3f} < {self.similarity_threshold}"
            )
            return decision

        if speaker_id and speaker_id in self.speaker_map:
            decision["status"] = "accepted"
            decision["allow_chat"] = True
            decision["speaker_name"] = self.speaker_map[speaker_id]["name"]
            decision["reason"] = "speaker recognized"
            return decision

        decision["status"] = "unknown"
        decision["allow_chat"] = True
        decision["speaker_name"] = "未知说话人"
        decision["reason"] = f"unknown speaker id: {speaker_id}"
        return decision

    async def _register_master(self, audio_data: bytes, session_id: str) -> bool:
        """动态模式注册 master_speaker。"""
        if not self.register_url:
            logger.bind(tag=TAG).error("register_url 未配置")
            return False

        headers = self._build_headers()
        data = aiohttp.FormData()
        data.add_field("speaker_id", self.dynamic_master_speaker_id)
        data.add_field("file", audio_data, filename="audio.wav", content_type="audio/wav")
        timeout = aiohttp.ClientTimeout(total=10)
        begin = time.monotonic()

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.register_url, headers=headers, data=data
                ) as response:
                    if response.status in (200, 201):
                        elapsed = time.monotonic() - begin
                        logger.bind(tag=TAG).info(
                            "动态声纹注册成功: "
                            f"speaker_id={self.dynamic_master_speaker_id}, "
                            f"sample={self._dynamic_registered_samples + 1}/"
                            f"{self.dynamic_registration_required_samples}, "
                            f"session={session_id}, elapsed={elapsed:.3f}s"
                        )
                        return True
                    body = await response.text()
                    logger.bind(tag=TAG).error(
                        f"动态声纹注册失败: HTTP {response.status}, body={body[:200]}"
                    )
                    return False
        except asyncio.TimeoutError:
            logger.bind(tag=TAG).error("动态声纹注册超时")
            return False
        except Exception as e:
            logger.bind(tag=TAG).error(f"动态声纹注册异常: {e}")
            return False

    async def _identify_by_speaker_ids(
        self, audio_data: bytes, speaker_ids: list
    ) -> Dict[str, Any]:
        """调用 identify 接口返回原始识别结果。"""
        result = {
            "ok": False,
            "speaker_id": None,
            "score": 0.0,
            "reason": "",
        }
        if not self.identify_url:
            result["reason"] = "identify_url not configured"
            return result
        if not speaker_ids:
            result["reason"] = "speaker_ids is empty"
            return result

        headers = self._build_headers()
        data = aiohttp.FormData()
        data.add_field("speaker_ids", ",".join(speaker_ids))
        data.add_field("file", audio_data, filename="audio.wav", content_type="audio/wav")
        timeout = aiohttp.ClientTimeout(total=10)
        begin = time.monotonic()
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.identify_url, headers=headers, data=data
                ) as response:
                    if response.status == 200:
                        body = await response.json()
                        result["ok"] = True
                        result["speaker_id"] = body.get("speaker_id")
                        result["score"] = float(body.get("score", 0.0) or 0.0)
                        elapsed = time.monotonic() - begin
                        logger.bind(tag=TAG).info(f"声纹识别耗时: {elapsed:.3f}s")
                        return result
                    text = await response.text()
                    result["reason"] = (
                        f"identify http {response.status}, body={text[:200]}"
                    )
                    logger.bind(tag=TAG).error(f"声纹识别API错误: {result['reason']}")
                    return result
        except asyncio.TimeoutError:
            result["reason"] = "identify timeout"
            logger.bind(tag=TAG).error("声纹识别超时")
            return result
        except Exception as e:
            result["reason"] = f"identify exception: {e}"
            logger.bind(tag=TAG).error(f"声纹识别失败: {e}")
            return result

    async def identify_speaker(self, audio_data: bytes, session_id: str) -> Optional[str]:
        """兼容旧接口：仅返回 speaker_name，不返回鉴权决策。"""
        decision = await self.evaluate_voiceprint(audio_data, session_id)
        return decision.get("speaker_name")

