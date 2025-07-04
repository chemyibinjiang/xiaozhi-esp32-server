package xiaozhi.modules.agent.service;


import xiaozhi.modules.agent.dto.AgentVoicePrintSaveDTO;
import xiaozhi.modules.agent.dto.AgentVoicePrintUpdateDTO;
import xiaozhi.modules.agent.vo.AgentVoicePrintVO;

import java.util.List;

/**
 * 智能体声纹处理service
 *
 * @author zjy
 */
public interface AgentVoicePrintService {
    /**
     * 添加智能体新的声纹
     *
     * @param dto 保存智能体声纹的数据
     * @return T:成功 F：失败
     */
    boolean insert(AgentVoicePrintSaveDTO dto);

    /**
     * 删除智能体的指的声纹
     *
     * @param voicePrintId 声纹id
     * @return 是否成功 T:成功 F：失败
     */
    boolean delete(String voicePrintId);

    /**
     * 获取指定智能体的所有声纹数据
     *
     * @param agentId 智能体id
     * @return 声纹数据集合
     */
    List<AgentVoicePrintVO> list(String agentId);

    /**
     * 更新智能体的指的声纹数据
     *
     * @param dto 修改的声纹的数据
     * @return 是否成功 T:成功 F：失败
     */
    boolean update(AgentVoicePrintUpdateDTO dto);



}
