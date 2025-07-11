package xiaozhi.modules.agent.dao;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;
import xiaozhi.modules.agent.entity.AgentChatHistoryEntity;
import xiaozhi.modules.agent.entity.AgentVoicePrintEntity;

/**
 * {@link AgentChatHistoryEntity} 智能体聊天历史记录Dao对象
 *
 * @author Goody
 * @version 1.0, 2025/4/30
 * @since 1.0.0
 */
@Mapper
public interface AgentVoicePrintDao extends BaseMapper<AgentVoicePrintEntity> {

}
