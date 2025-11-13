-- 当删除知识库记录时，自动删除相关的插件映射记录

-- 先删除可能存在的同名触发器（如果存在）
DROP TRIGGER IF EXISTS trigger_delete_plugin_mapping_on_knowledgebase_delete;

-- 创建新的触发器
CREATE TRIGGER trigger_delete_plugin_mapping_on_knowledgebase_delete
AFTER DELETE ON ai_rag_dataset
FOR EACH ROW
BEGIN
    -- 删除与该知识库ID相关的插件映射记录
    DELETE FROM ai_agent_plugin_mapping WHERE plugin_id = OLD.id;
END;
