-- OpenAI ASR模型供应器
delete from `ai_model_provider` where id = 'SYSTEM_ASR_GptASR';
INSERT INTO `ai_model_provider` (`id`, `model_type`, `provider_code`, `name`, `fields`, `sort`, `creator`, `create_date`, `updater`, `update_date`) VALUES
('SYSTEM_ASR_GptASR', 'ASR', 'gpt', 'Gpt语音识别', '[{"key": "base_url", "type": "string", "label": "基础URL"}, {"key": "model_name", "type": "string", "label": "模型名称"}, {"key": "api_key", "type": "string", "label": "API密钥"}, {"key": "output_dir", "type": "string", "label": "输出目录"}]', 5, 1, NOW(), 1, NOW());


-- OpenAI ASR模型配置
delete from `ai_model_config` where id = 'ASR_GptASR';
INSERT INTO `ai_model_config` VALUES ('ASR_GptASR', 'ASR', 'GptASR', 'GPT语音识别', 0, 1, '{\"type\": \"gpt\", \"api_key\": \"\", \"base_url\": \"https://api.openai.com/v1/audio/transcriptions\", \"model_name\": \"gpt-4o-mini-transcribe\", \"output_dir\": \"tmp/\"}', NULL, NULL, 3, NULL, NULL, NULL, NULL);


-- 更新OpenAI ASR配置说明
UPDATE `ai_model_config` SET 
`doc_link` = 'https://platform.openai.com/docs/api-reference/audio/createTranscription',
`remark` = 'OpenAI ASR配置说明：
1. 需要在OpenAI开放平台创建组织并获取api_key
2. 支持中、英、日、韩等多种语音识别，具体参考文档https://platform.openai.com/docs/guides/speech-to-text
3. 需要网络连接
4. 输出文件保存在tmp/目录
5. 兼容groq语音识别。 groq识别速度更快。具体参考文档：https://console.groq.com/docs/speech-to-text
申请步骤：
**OpenAi ASR申请步骤：**
1.登录OpenAI Platform。https://auth.openai.com/log-in
2.创建api-key  https://platform.openai.com/settings/organization/api-keys
3.模型可以选择gpt-4o-transcribe或GPT-4o mini Transcribe
**groq ASR申请步骤：**
1.登录groq Console。https://console.groq.com/home
2.创建api-key  https://console.groq.com/keys
3.模型可以选择whisper-large-v3-turbo或whisper-large-v3（distil-whisper-large-v3-en仅支持英语转录）
4.base url https://api.groq.com/openai/v1/audio/transcriptions
' WHERE `id` = 'ASR_GptASR';