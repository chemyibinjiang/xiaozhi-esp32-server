# PaddleSpeechTTS集成xiaozhi服务

## 一、基础环境要求
操作系统：Windows / Linux / WSL 2

Python 版本：3.8 ~ 3.10

Paddle 官方最新版本

依赖管理工具：conda 或 venv

前提条件：需要先根据paddle官方教程安装好PaddlePaddle   ```https://www.paddlepaddle.org.cn/install```

## 二、启动paddlespeech服务
### 2.1从paddlespeech官方仓库拉取源码
```bash 
git clone https://github.com/PaddlePaddle/PaddleSpeech.git
```
### 2.2建立虚拟环境
```bash
conda create -n paddle_env python=3.8 -y
conda activate paddle_env
```
### 2.3进入paddlespeech目录
```bash
cd PaddleSpeech
```
### 2.4安装paddlespeech
```bash
pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple

#以下命令使用任意一个
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
pip install paddlespeech -i https://pypi.tuna.tsinghua.edu.cn/simple
```
### 2.5使用命令自动下载语音模型
```bash
paddlespeech tts --input "你好，这是一次测试"
```
此步骤会自动下载模型缓存至本地 .paddlespeech/models 目录

### 2.6修改tts_online_application.yaml配置
参考目录 ```"PaddleSpeech\demos\streaming_tts_server\conf\tts_online_application.yaml"```
选择```tts_online_application.yaml```文件用编辑器打开，设置```protocol```为```websocket```

### 2.7启动服务
```yaml
paddlespeech_server start --config_file ./demos/streaming_tts_server/conf/tts_online_application.yaml
```
请根据你的```tts_online_application.yaml```的实际目录来启动命令
看到如下日志即启动成功
```
Prefix dict has been built successfully.
[2025-08-07 10:03:11,312] [   DEBUG] __init__.py:166 - Prefix dict has been built successfully.
INFO:     Started server process [2298]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8092 (Press CTRL+C to quit)
```

## 三、修改小智的配置文件
### 3.1```main/xiaozhi-server/core/providers/tts/paddle_speech.py```
```py
import asyncio
import json
import base64
import aiohttp
import numpy as np
import io
import wave
import websockets
from core.providers.tts.base import TTSProviderBase
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)
        self.url = config.get("url", "ws://192.168.1.10:8092/paddlespeech/tts/streaming")
        self.protocol = config.get("protocol", "websocket")
        self.spk_id = config.get("spk_id", 0)
        self.sample_rate = config.get("sample_rate", 24000)
        self.speed = config.get("speed", 1.0)
        self.volume = config.get("volume", 1.0)
        self.save_path = config.get("save_path", "./streaming_tts.wav")

    async def pcm_to_wav(self, pcm_data: bytes, sample_rate: int = 24000, num_channels: int = 1,
                         bits_per_sample: int = 16) -> bytes:
        """
        将 PCM 数据转换为 WAV 文件并返回字节数据
        :param pcm_data: PCM 数据（原始字节流）
        :param sample_rate: 音频采样率，默认为24000
        :param num_channels: 声道数，默认为单声道
        :param bits_per_sample: 每个样本的位数，默认为16
        :return: WAV 格式的字节数据
        """
        byte_data = np.frombuffer(pcm_data, dtype=np.int16)  # 16位PCM
        wav_io = io.BytesIO()

        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(bits_per_sample // 8)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(byte_data.tobytes())

        return wav_io.getvalue()

    async def text_to_speak(self, text, output_file):
        if self.protocol == "websocket":
            return await self.text_streaming(text, output_file)
        elif self.protocol == "http":
            return await self.text(text, output_file)
        else:
            raise ValueError("Unsupported protocol. Please use 'websocket' or 'http'.")

    async def text(self, text, output_file):
        request_json = {
            "text": text,
            "spk_id": self.spk_id,
            "speed": self.speed,
            "volume": self.volume,
            "sample_rate": self.sample_rate,
            "save_path": self.save_path
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, json=request_json) as resp:
                    if resp.status == 200:
                        resp_json = await resp.json()
                        if resp_json.get("success"):
                            data = resp_json["result"]
                            audio_bytes = base64.b64decode(data["audio"])
                            if output_file:
                                with open(output_file, "wb") as file_to_save:
                                    file_to_save.write(audio_bytes)
                            else:
                                return audio_bytes
                        else:
                            raise Exception(
                                f"Error: {resp_json.get('message', 'Unknown error')} while processing text: {text}")
                    else:
                        raise Exception(
                            f"HTTP Error: {resp.status} - {await resp.text()} while processing text: {text}")
        except Exception as e:
            raise Exception(f"Error during TTS HTTP request: {e} while processing text: {text}")

    async def text_streaming(self, text, output_file):
        try:
            # 使用 websockets 异步连接到 WebSocket 服务器
            async with websockets.connect(self.url) as ws:
                # 发送开始请求
                start_request = {
                    "task": "tts",
                    "signal": "start"
                }
                await ws.send(json.dumps(start_request))

                # 接收开始响应并提取 session_id
                start_response = await ws.recv()
                start_response = json.loads(start_response)  # 解析 JSON 响应
                if start_response.get("status") != 0:
                    raise Exception(f"连接失败: {start_response.get('signal')}")

                session_id = start_response.get("session")

                # 发送待合成的文本数据
                data_request = {
                    "text": text,
                    "spk_id": self.spk_id,
                }
                await ws.send(json.dumps(data_request))

                audio_chunks = b""
                timeout_seconds = 60  # 设置超时
                try:
                    while True:
                        response = await asyncio.wait_for(ws.recv(), timeout=timeout_seconds)
                        response = json.loads(response)  # 解析 JSON 响应
                        status = response.get("status")

                        if status == 2:  # 最后一个数据包
                            break
                        else:
                            # 拼接音频数据（base64 编码的 PCM 数据）
                            audio_chunks += base64.b64decode(response.get("audio"))
                except asyncio.TimeoutError:
                    raise Exception(f"WebSocket 超时：等待音频数据超过 {timeout_seconds} 秒")

                # 将拼接后的 PCM 数据转换为 WAV 格式
                wav_data = await self.pcm_to_wav(audio_chunks)

                # 结束请求
                end_request = {
                    "task": "tts",
                    "signal": "end",
                    "session": session_id  # 会话 ID 必须与开始请求中的一致
                }
                await ws.send(json.dumps(end_request))

                # 接收结束响应避免服务抛出异常
                await ws.recv()

                # 返回或保存音频数据
                if output_file:
                    with open(output_file, "wb") as file_to_save:
                        file_to_save.write(wav_data)
                else:
                    return wav_data

        except Exception as e:
            raise Exception(f"Error during TTS WebSocket request: {e} while processing text: {text}")
```
### 3.2```main/xiaozhi-server/data/.config.yaml```
使用单模块部署
```yaml
selected_module:
  TTS: PaddleSpeechTTS
TTS:
  PaddleSpeechTTS:
      type: paddle_speech
      protocol: websocket 
      url:  ws://127.0.0.1:8092/paddlespeech/tts/streaming  # TTS 服务的 URL 地址，指向本地服务器 [websocket默认ws://127.0.0.1:8092/paddlespeech/tts/streaming]
      spk_id: 0  # 发音人 ID，0 通常表示默认的发音人
      sample_rate: 24000  # 采样率 [websocket默认24000，http默认0 自动选择]
      speed: 1.0  # 语速，1.0 表示正常语速，>1 表示加快，<1 表示减慢
      volume: 1.0  # 音量，1.0 表示正常音量，>1 表示增大，<1 表示减小
      save_path: ./streaming_tts.wav  # 服务器生成的语音文件保存路径
```
### 3.3启动xiaozhi服务
```py
python app.py
```
打开test目录下的test_page.html，测试连接和发送消息时paddlespeech端是否有输出日志

输出日志参考：
```b
INFO:     127.0.0.1:44312 - "WebSocket /paddlespeech/tts/streaming" [accepted]
INFO:     connection open
[2025-08-07 11:16:33,355] [    INFO] - sentence: 哈哈，怎么突然找我聊天啦？
[2025-08-07 11:16:33,356] [    INFO] - The durations of audio is: 2.4625 s
[2025-08-07 11:16:33,356] [    INFO] - first response time: 0.1143045425415039 s
[2025-08-07 11:16:33,356] [    INFO] - final response time: 0.4777836799621582 s
[2025-08-07 11:16:33,356] [    INFO] - RTF: 0.19402382942625715
[2025-08-07 11:16:33,356] [    INFO] - Other info: front time: 0.06514096260070801 s, first am infer time: 0.008037090301513672 s, first voc infer time: 0.04112648963928223 s,
[2025-08-07 11:16:33,356] [    INFO] - Complete the synthesis of the audio streams
INFO:     connection closed

```
