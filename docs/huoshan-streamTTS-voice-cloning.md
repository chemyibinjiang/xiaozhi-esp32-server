# 火山双流式语音合成+音色克隆配置教程
使用火山引擎双向流式语音合成服务的同时进行声音克隆，支持WebSocket协议流式调用。
### 1.开通火山引擎服务
访问 https://console.volcengine.com/speech/app  在应用管理创建应用，勾选语音合成大模型和声音复刻大模型，左边列表点击声音复刻大模型后下滑获得App Id，Access Token，Cluster ID以及声音ID(S_xxxxx)
### 2.克隆音色
#####2.1全模块部署
1.填写配置。
如果你是全模块部署，请在智控台页面模型配置的语音合成页面，找到“火山双流式语音合成”，点击修改，将你火山引擎的App Id，Access Token以及声音ID(S_xxxxx)填入，注意：把资源id（Resource-Id）改成```volc.megatts.defaul```或```seed-icl-1.0```

2.音色克隆。
准备一段8-60之间的音频文件，点击音色克隆菜单的音色资源，新增音色资源。回到音色克隆页面，点击上传音频。上传好音频之后点击复刻，等1~2秒会返回结果。
复刻成功后，在“火山双流式语音合成”的“音色资源”里可以看到你复刻好的声音，也可以在配置角色使用“火山双流式语音合成”时选择复刻的音色了。

#####2.2单模块部署
克隆音色请参照教程 https://github.com/104gogo/huoshan-voice-copy 

1.音色克隆。
准备一段 10-30 秒的音频文件（.wav格式）添加到克隆的项目中，将平台获得的密钥填入```uploadAndStatus.py```和```tts_http_demo.py```

在uploadAndStatus.py中，将 audio_path=修改成自己的.wav文件名称
```python
train(appid=appid, token=token, audio_path=r".\audios\xiaohe.wav", spk_id=spk_id)
```

运行以下命令生成test_submit.mp3，点击播放试听克隆效果

```python
python uploadAndStatus.py
python tts_http_demo.py
```
回到火山引擎控制台页面，刷新可以看到声音复刻详情的状态是复刻成功。

2.填写配置文件（.config.yaml）
修改 resource_id的参数为``` volc.megatts.default``` 
speaker的参数填入声音ID(S_xxxxx)

启动服务，唤醒小智发出的声音是克隆的音色即成功。