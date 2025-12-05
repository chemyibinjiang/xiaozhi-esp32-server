-- 添加系统功能菜单配置参数
delete from `sys_params` where param_code = 'system-web.menu';

-- 添加系统功能菜单配置参数
INSERT INTO `sys_params` (id, param_code, param_value, value_type, param_type, remark) VALUES 
(600, 'system-web.menu', '{"voiceprintRecognition":{"name":"feature.voiceprintRecognition.name","enabled":false,"description":"feature.voiceprintRecognition.description"},"voiceClone":{"name":"feature.voiceClone.name","enabled":false,"description":"feature.voiceClone.description"},"knowledgeBase":{"name":"feature.knowledgeBase.name","enabled":false,"description":"feature.knowledgeBase.description"},"mcpAccessPoint":{"name":"feature.mcpAccessPoint.name","enabled":false,"description":"feature.mcpAccessPoint.description"},"vad":{"name":"feature.vad.name","enabled":false,"description":"feature.vad.description"},"asr":{"name":"feature.asr.name","enabled":false,"description":"feature.asr.description"}}', 'json', 1, '系统功能菜单配置');