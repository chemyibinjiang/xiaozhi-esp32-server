delete from `sys_params` where param_code = 'mqtt.signature_key';
INSERT INTO `sys_params` (id, param_code, param_value, value_type, param_type, remark) VALUES (120, 'mqtt.signature_key', 'null', 'string', 1, 'mqtt 密钥 配置');
