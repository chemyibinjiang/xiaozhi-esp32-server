import uuid
import json
import hmac
import hashlib
import base64
import time
import queue
import asyncio
import traceback
from asyncio import Task
import websockets
import os
from datetime import datetime
from urllib import parse
from core.providers.tts.base import TTSProviderBase
from core.providers.tts.dto.dto import SentenceType, ContentType, InterfaceType
from core.utils.tts import MarkdownCleaner
from core.utils import opus_encoder_utils
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


class AccessToken:
    @staticmethod
    def _encode_text(text):
        encoded_text = parse.quote_plus(text)
        return encoded_text.replace("+", "%20").replace("*", "%2A").replace("%7E", "~")

    @staticmethod
    def _encode_dict(dic):
        keys = dic.keys()
        dic_sorted = [(key, dic[key]) for key in sorted(keys)]
        encoded_text = parse.urlencode(dic_sorted)
        return encoded_text.replace("+", "%20").replace("*", "%2A").replace("%7E", "~")

    @staticmethod
    def create_token(access_key_id, access_key_secret):
        parameters = {
            "AccessKeyId": access_key_id,
            "Action": "CreateToken",
            "Format": "JSON",
            "RegionId": "cn-shanghai",  # 使用上海地域进行Token获取
            "SignatureMethod": "HMAC-SHA1",
            "SignatureNonce": str(uuid.uuid1()),
            "SignatureVersion": "1.0",
            "Timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "Version": "2019-02-28",
        }

        query_string = AccessToken._encode_dict(parameters)
        string_to_sign = (
            "GET"
            + "&"
            + AccessToken._encode_text("/")
            + "&"
            + AccessToken._encode_text(query_string)
        )

        secreted_string = hmac.new(
            bytes(access_key_secret + "&", encoding="utf-8"),
            bytes(string_to_sign, encoding="utf-8"),
            hashlib.sha1,
        ).digest()
        signature = base64.b64encode(secreted_string)
        signature = AccessToken._encode_text(signature)

        full_url = "http://nls-meta.cn-shanghai.aliyuncs.com/?Signature=%s&%s" % (
            signature,
            query_string,
        )

        import requests
        response = requests.get(full_url)
        if response.ok:
            root_obj = response.json()
            key = "Token"
            if key in root_obj:
                token = root_obj[key]["Id"]
                expire_time = root_obj[key]["ExpireTime"]
                return token, expire_time
        return None, None


class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)

        # 设置为流式接口类型
        self.interface_type = InterfaceType.DUAL_STREAM

        # 基础配置
        self.access_key_id = config.get("access_key_id")
        self.access_key_secret = config.get("access_key_secret")
        self.appkey = config.get("appkey")
        self.format = config.get("format", "pcm")
        self.audio_file_type = config.get("format", "pcm")

        # 采样率配置
        sample_rate = config.get("sample_rate", "16000")
        self.sample_rate = int(sample_rate) if sample_rate else 16000

        # 音色配置 - CosyVoice大模型音色
        if config.get("private_voice"):
            self.voice = config.get("private_voice")
        else:
            self.voice = config.get("voice", "longxiaochun")  # CosyVoice默认音色

        # 音频参数配置
        volume = config.get("volume", "50")
        self.volume = int(volume) if volume else 50

        speech_rate = config.get("speech_rate", "0")
        self.speech_rate = int(speech_rate) if speech_rate else 0

        pitch_rate = config.get("pitch_rate", "0")
        self.pitch_rate = int(pitch_rate) if pitch_rate else 0

        # WebSocket配置
        self.host = config.get("host", "nls-gateway-cn-beijing.aliyuncs.com")
        self.ws_url = f"wss://{self.host}/ws/v1"
        self.ws = None
        self._monitor_task = None

        # 文本获取
        self.sentence_queue = queue.Queue()
        self.text_buffer = ""
        self.sentence_end_chars = {'.', '。', '!', '！', '?', '？', '\n', "~", "；", ";", ":", "：", " ", "，", ","}

        # 创建Opus编码器
        self.opus_encoder = opus_encoder_utils.OpusEncoderUtils(
            sample_rate=16000, channels=1, frame_size_ms=60
        )

        # PCM缓冲区
        self.pcm_buffer = bytearray()

        # Token管理
        if self.access_key_id and self.access_key_secret:
            self._refresh_token()
        else:
            self.token = config.get("token")
            self.expire_time = None

    def _refresh_token(self):
        """刷新Token并记录过期时间"""
        if self.access_key_id and self.access_key_secret:
            self.token, expire_time_str = AccessToken.create_token(
                self.access_key_id, self.access_key_secret
            )
            if not expire_time_str:
                raise ValueError("无法获取有效的Token过期时间")

            expire_str = str(expire_time_str).strip()

            try:
                if expire_str.isdigit():
                    expire_time = datetime.fromtimestamp(int(expire_str))
                else:
                    expire_time = datetime.strptime(expire_str, "%Y-%m-%dT%H:%M:%SZ")
                self.expire_time = expire_time.timestamp() - 60
            except Exception as e:
                raise ValueError(f"无效的过期时间格式: {expire_str}") from e
        else:
            self.expire_time = None

        if not self.token:
            raise ValueError("无法获取有效的访问Token")

    def _is_token_expired(self):
        """检查Token是否过期"""
        if not self.expire_time:
            return False
        return time.time() > self.expire_time

    async def _ensure_connection(self):
        """确保WebSocket连接可用"""
        try:
            if self._is_token_expired():
                logger.bind(tag=TAG).warning("Token已过期，正在自动刷新...")
                self._refresh_token()
            if self.ws:
                # 10秒内才可以复用,适合连续对话
                logger.bind(tag=TAG).info(f"使用已有链接...")
                return self.ws
            logger.bind(tag=TAG).info("开始建立新连接...")

            self.ws = await websockets.connect(
                self.ws_url,
                additional_headers={"X-NLS-Token": self.token},
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10,
            )
            logger.bind(tag=TAG).info("WebSocket连接建立成功")
            return self.ws
        except Exception as e:
            logger.bind(tag=TAG).error(f"建立连接失败: {str(e)}")
            self.ws = None
            raise

    def tts_text_priority_thread(self):
        """流式文本处理线程"""
        while not self.conn.stop_event.is_set():
            try:
                message = self.tts_text_queue.get(timeout=1)
                logger.bind(tag=TAG).debug(
                    f"收到TTS任务｜{message.sentence_type.name} ｜ {message.content_type.name} | 会话ID: {self.conn.sentence_id}"
                )

                if message.sentence_type == SentenceType.FIRST:
                    self.conn.client_abort = False

                if self.conn.client_abort:
                    logger.bind(tag=TAG).info("收到打断信息，终止TTS文本处理线程")
                    continue

                if message.sentence_type == SentenceType.FIRST:
                    # 初始化参数
                    try:
                        if not getattr(self.conn, "sentence_id", None):
                            self.conn.sentence_id = uuid.uuid4().hex
                            logger.bind(tag=TAG).info(f"自动生成新的 会话ID: {self.conn.sentence_id}")

                        # aliyun独有的message_id需要自己生成
                        self.conn.message_id = str(uuid.uuid4().hex)

                        logger.bind(tag=TAG).info("开始启动TTS会话...")
                        future = asyncio.run_coroutine_threadsafe(
                            self.start_session(self.conn.sentence_id),
                            loop=self.conn.loop,
                        )
                        future.result()
                        self.before_stop_play_files.clear()
                        logger.bind(tag=TAG).info("TTS会话启动成功")

                    except Exception as e:
                        logger.bind(tag=TAG).error(f"启动TTS会话失败: {str(e)}")
                        continue

                elif ContentType.TEXT == message.content_type:
                    if message.content_detail:
                        try:
                            logger.bind(tag=TAG).debug(
                                f"开始发送TTS文本: {message.content_detail}"
                            )
                            self.text_buffer += message.content_detail
                            if message.content_detail in self.sentence_end_chars and len(self.text_buffer) > 6:
                                self.sentence_queue.put(self.text_buffer)
                                self.text_buffer = ""
                            future = asyncio.run_coroutine_threadsafe(
                                self.text_to_speak(message.content_detail, None),
                                loop=self.conn.loop,
                            )
                            future.result()
                            logger.bind(tag=TAG).debug("TTS文本发送成功")
                        except Exception as e:
                            logger.bind(tag=TAG).error(f"发送TTS文本失败: {str(e)}")
                            continue

                elif ContentType.FILE == message.content_type:
                    logger.bind(tag=TAG).info(
                        f"添加音频文件到待播放列表: {message.content_file}"
                    )
                    if message.content_file and os.path.exists(message.content_file):
                        # 先处理文件音频数据
                        file_audio = self._process_audio_file(message.content_file)
                        self.before_stop_play_files.append(
                            (file_audio, message.content_detail)
                        )

                if message.sentence_type == SentenceType.LAST:
                    try:
                        logger.bind(tag=TAG).info("开始结束TTS会话...")
                        self.sentence_queue.put(self.text_buffer)
                        print(list(self.sentence_queue.queue))
                        future = asyncio.run_coroutine_threadsafe(
                            self.finish_session(self.conn.sentence_id),
                            loop=self.conn.loop,
                        )
                        future.result()
                    except Exception as e:
                        logger.bind(tag=TAG).error(f"结束TTS会话失败: {str(e)}")
                        continue

            except queue.Empty:
                continue
            except Exception as e:
                logger.bind(tag=TAG).error(
                    f"处理TTS文本失败: {str(e)}, 类型: {type(e).__name__}, 堆栈: {traceback.format_exc()}"
                )

    async def text_to_speak(self, text, _):
        try:
            if self.ws is None:
                logger.bind(tag=TAG).warning(f"WebSocket连接不存在，终止发送文本")
                return
            filtered_text = MarkdownCleaner.clean_markdown(text)
            run_request = {
                "header": {
                    "message_id": self.conn.message_id,
                    "task_id": self.conn.sentence_id,
                    "namespace": "FlowingSpeechSynthesizer",
                    "name": "RunSynthesis",
                    "appkey": self.appkey,
                },
                "payload": {
                    "text": filtered_text
                }
            }
            await self.ws.send(json.dumps(run_request))
            return

        except Exception as e:
            logger.bind(tag=TAG).error(f"发送TTS文本失败: {str(e)}")
            if self.ws:
                try:
                    await self.ws.close()
                except:
                    pass
                self.ws = None
            raise

    async def start_session(self, session_id):
        logger.bind(tag=TAG).info(f"开始会话～～{session_id}")
        try:
            # 会话开始时检测上个会话的监听状态
            if(
                self._monitor_task is not None
                and isinstance(self._monitor_task, Task)
                and not self._monitor_task.done()
            ):
                logger.bind(tag=TAG).info("检测到未完成的上个会话，关闭监听任务和连接...")
                await self.close()

            # 建立新连接
            await self._ensure_connection()

            # 启动监听任务
            self._monitor_task = asyncio.create_task(self._start_monitor_tts_response())

            start_request = {
                "header": {
                    "message_id": self.conn.message_id,
                    "task_id": self.conn.sentence_id,
                    "namespace": "FlowingSpeechSynthesizer",
                    "name": "StartSynthesis",
                    "appkey": self.appkey,
                },
                "payload": {
                    "voice": self.voice,
                    "format": self.format,
                    "sample_rate": self.sample_rate,
                    "volume": self.volume,
                    "speech_rate": self.speech_rate,
                    "pitch_rate": self.pitch_rate,
                    "enable_subtitle": True
                }
            }
            await self.ws.send(json.dumps(start_request))
            logger.bind(tag=TAG).info("会话启动请求已发送")
        except Exception as e:
            logger.bind(tag=TAG).error(f"启动会话失败: {str(e)}")
            # 确保清理资源
            await self.close()
            raise

    async def finish_session(self, session_id):
        logger.bind(tag=TAG).info(f"关闭会话～～{session_id}")
        try:
            if self.ws:
                stop_request = {
                    "header": {
                        "message_id": self.conn.message_id,
                        "task_id": self.conn.sentence_id,
                        "namespace": "FlowingSpeechSynthesizer",
                        "name": "StopSynthesis",
                        "appkey": self.appkey,
                    }
                }
                await self.ws.send(json.dumps(stop_request))
                logger.bind(tag=TAG).info("会话结束请求已发送")
                if self._monitor_task:
                    try:
                        await self._monitor_task
                    except Exception as e:
                        logger.bind(tag=TAG).error(
                            f"等待监听任务完成时发生错误: {str(e)}"
                        )
                    finally:
                        self._monitor_task = None
        except Exception as e:
            logger.bind(tag=TAG).error(f"关闭会话失败: {str(e)}")
            # 确保清理资源
            await self.close()
            raise

    async def close(self):
        """资源清理"""
        if self._monitor_task:
            try:
                self._monitor_task.cancel()
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.bind(tag=TAG).warning(f"关闭时取消监听任务错误: {e}")
            self._monitor_task = None

        if self.ws:
            try:
                await self.ws.close()
            except:
                pass
            self.ws = None

    async def _start_monitor_tts_response(self):
        """监听TTS响应"""
        opus_datas_cache = []
        is_first_sentence = True
        first_sentence_segment_count = 0  # 添加计数器
        text_buff = ""
        try:
            session_finished = False  # 标记会话是否正常结束
            while not self.conn.stop_event.is_set():
                try:
                    msg = await self.ws.recv()
                    # 检查客户端是否中止
                    if self.conn.client_abort:
                        logger.bind(tag=TAG).info("收到打断信息，终止监听TTS响应")
                        break
                    if isinstance(msg, str):  # 文本控制消息
                        try:
                            data = json.loads(msg)
                            header = data.get("header", {})
                            event_name = header.get("name")
                            if event_name == "SynthesisStarted":
                                logger.bind(tag=TAG).debug("TTS合成已启动")
                            elif event_name == "SentenceBegin":
                                try:
                                    text_buff = self.sentence_queue.get_nowait()
                                except queue.Empty:
                                    text_buff = ""
                                logger.bind(tag=TAG).debug(f"句子语音生成开始: {text_buff}")
                                opus_datas_cache = []
                                self.tts_audio_queue.put((SentenceType.FIRST, [], text_buff))
                            elif event_name == "SentenceEnd":
                                logger.bind(tag=TAG).info(f"句子语音生成成功： {text_buff}")
                                text_buff = ""
                                if not is_first_sentence or first_sentence_segment_count > 10:
                                    # 发送缓存的数据
                                    self.tts_audio_queue.put(
                                        (SentenceType.MIDDLE, opus_datas_cache, None)
                                    )
                                # 第一句话结束后，将标志设置为False
                                is_first_sentence = False
                            elif event_name == "SynthesisCompleted":
                                logger.bind(tag=TAG).debug(f"会话结束～～")
                                self._process_before_stop_play_files()
                                session_finished = True
                                break
                        except json.JSONDecodeError:
                            logger.bind(tag=TAG).warning("收到无效的JSON消息")
                    # 二进制消息（音频数据）
                    elif isinstance(msg, (bytes, bytearray)):
                        logger.bind(tag=TAG).debug(f"推送数据到队列里面～～")
                        opus_datas = self.opus_encoder.encode_pcm_to_opus(msg, False)
                        logger.bind(tag=TAG).debug(
                            f"推送数据到队列里面帧数～～{len(opus_datas)}"
                        )
                        if is_first_sentence:
                            first_sentence_segment_count += 1
                            if first_sentence_segment_count <= 6:
                                self.tts_audio_queue.put(
                                    (SentenceType.MIDDLE, opus_datas, None)
                                )
                            else:
                                opus_datas_cache = opus_datas_cache + opus_datas
                        else:
                            # 后续句子缓存
                            opus_datas_cache = opus_datas_cache + opus_datas

                except websockets.ConnectionClosed:
                    logger.bind(tag=TAG).warning("WebSocket连接已关闭")
                    break
                except Exception as e:
                    logger.bind(tag=TAG).error(
                        f"处理TTS响应时出错: {e}\n{traceback.format_exc()}"
                    )
                    break
            # 仅在连接异常时才关闭
            if not session_finished and self.ws:
                try:
                    await self.ws.close()
                except:
                    pass
                self.ws = None
        # 监听任务退出时清理引用
        finally:
            self._monitor_task = None

    def to_tts(self, text: str) -> list:
        """非流式TTS处理，用于测试及保存音频文件的场景"""
        start_time = time.time()
        text = MarkdownCleaner.clean_markdown(text)

        try:
            # 使用同步方式进行TTS转换
            if self._is_token_expired():
                self._refresh_token()

            # 构造请求数据
            request_json = {
                "appkey": self.appkey,
                "token": self.token,
                "text": text,
                "format": "pcm",
                "sample_rate": self.sample_rate,
                "voice": self.voice,
                "volume": self.volume,
                "speech_rate": self.speech_rate,
                "pitch_rate": self.pitch_rate,
            }

            # 使用HTTP接口进行同步请求
            import requests
            api_url = f"https://{self.host}/stream/v1/tts"
            headers = {"Content-Type": "application/json"}

            resp = requests.post(api_url, json=request_json, headers=headers)

            if resp.status_code == 401:  # Token过期特殊处理
                self._refresh_token()
                resp = requests.post(api_url, json=request_json, headers=headers)

            if resp.headers["Content-Type"].startswith("audio/"):
                pcm_data = resp.content

                # 使用opus编码器处理PCM数据
                opus_datas = []
                frame_bytes = int(
                    self.opus_encoder.sample_rate
                    * self.opus_encoder.channels
                    * self.opus_encoder.frame_size_ms
                    / 1000
                    * 2
                )

                # 分帧处理PCM数据
                for i in range(0, len(pcm_data), frame_bytes):
                    frame = pcm_data[i:i + frame_bytes]
                    if len(frame) == frame_bytes:
                        opus = self.opus_encoder.encode_pcm_to_opus(frame, False)
                        if opus:
                            opus_datas.extend(opus)

                logger.bind(tag=TAG).info(f"TTS请求成功: {text}, 耗时: {time.time() - start_time}秒")
                return opus_datas
            else:
                logger.bind(tag=TAG).error(f"TTS请求失败: {resp.content}")
                return []

        except Exception as e:
            logger.bind(tag=TAG).error(f"TTS请求异常: {e}")
            return []
