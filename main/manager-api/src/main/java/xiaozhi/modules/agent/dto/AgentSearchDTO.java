package xiaozhi.modules.agent.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description = "智能体搜索查询DTO")
public class AgentSearchDTO {
    @Schema(description = "搜索关键词")
    private String keyword;

    @Schema(description = "搜索类型: mac - 按MAC地址搜索, name - 按名称搜索")
    private String searchType;

    @Schema(description = "用户ID")
    private Long userId;
}
