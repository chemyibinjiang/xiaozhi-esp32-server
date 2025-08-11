-- 更新HuoshanDoubleStreamTTS供应器增加语速，音调等配置
UPDATE `ai_model_provider`
SET fields = '[{"key": "ws_url", "type": "string", "label": "WebSocket地址"}, {"key": "appid", "type": "string", "label": "应用ID"}, {"key": "access_token", "type": "string", "label": "访问令牌"}, {"key": "resource_id", "type": "string", "label": "资源ID"}, {"key": "speaker", "type": "string", "label": "默认音色"}, {"key": "speech_rate", "type": "number", "label": "语速(-50~100)"}, {"key": "loudness_rate", "type": "number", "label": "音量(-50~100)"}, {"key": "pitch", "type": "number", "label": "音高(-12~12)"}]'
WHERE id = 'SYSTEM_TTS_HSDSTTS';