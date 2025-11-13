UPDATE `ai_model_config` SET 
`doc_link` = 'https://github.com/infiniflow/ragflow/blob/main/README_zh.md',
`remark` = '详细部署教程参考：https://github.com/xinnan-tech/xiaozhi-esp32-server/blob/main/docs/ragflow-integration.md
部署成功，注冊登录后，点击右上角头像，获得RAGFlow的API KEY和API服务器地址。使用RAGFlow前请在Model Provider中添加模型和设置默认模型。' WHERE `id` = 'RAG_RAGFlow';

update `ai_model_config` set `model_name` = '外挂的大模型意图识别' where `id` = 'Intent_intent_llm';
update `ai_model_config` set `model_name` = '大模型自主函数调用' where `id` = 'Intent_function_call';

update `ai_model_provider` set `name` = '外挂的大模型意图识别' , `fields` =  '[{"key":"llm","label":"引用的LLM模型","type":"string"}]' where  id = 'SYSTEM_Intent_intent_llm';
update `ai_model_provider` set `name` = '大模型自主函数调用' where  id = 'SYSTEM_Intent_function_call';

