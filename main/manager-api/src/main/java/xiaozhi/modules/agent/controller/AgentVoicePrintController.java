package xiaozhi.modules.agent.controller;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.AllArgsConstructor;
import org.apache.shiro.authz.annotation.RequiresPermissions;
import org.springframework.web.bind.annotation.*;
import xiaozhi.common.utils.Result;
import xiaozhi.modules.agent.dto.AgentVoicePrintSaveDTO;
import xiaozhi.modules.agent.dto.AgentVoicePrintUpdateDTO;
import xiaozhi.modules.agent.service.AgentVoicePrintService;
import xiaozhi.modules.agent.vo.AgentVoicePrintVO;
import xiaozhi.modules.security.user.SecurityUser;

import java.util.List;

@Tag(name = "智能体声纹管理")
@AllArgsConstructor
@RestController
@RequestMapping("/agent/voice-print")
public class AgentVoicePrintController {
    private final AgentVoicePrintService agentVoicePrintService;

    @PostMapping
    @Operation(summary = "创建智能体的声纹")
    @RequiresPermissions("sys:role:normal")
    public Result<Void> save(@RequestBody @Valid AgentVoicePrintSaveDTO dto) {
        boolean b = agentVoicePrintService.insert(dto);
        if (b) {
            return new Result<>();
        }
        return new Result<Void>().error("智能体的声纹创建失败");
    }

    @PutMapping
    @Operation(summary = "更新智能体的对应声纹")
    @RequiresPermissions("sys:role:normal")
    public Result<Void> update(@RequestBody @Valid AgentVoicePrintUpdateDTO dto) {
        Long userId = SecurityUser.getUserId();
        boolean b = agentVoicePrintService.update(userId, dto);
        if (b) {
            return new Result<>();
        }
        return new Result<Void>().error("智能体的对应声纹更新失败");
    }

    @DeleteMapping("/{id}")
    @Operation(summary = "删除智能体对应声纹")
    @RequiresPermissions("sys:role:normal")
    public Result<Void> delete(@PathVariable String id) {
        Long userId = SecurityUser.getUserId();
        // 先删除关联的设备
        boolean delete = agentVoicePrintService.delete(userId, id);
        if (delete) {
            return new Result<>();
        }
        return new Result<Void>().error("智能体的对应声纹删除失败");
    }

    @GetMapping("/list/{id}")
    @Operation(summary = "获取用户指定智能体声纹列表")
    @RequiresPermissions("sys:role:normal")
    public Result<List<AgentVoicePrintVO>> list(@PathVariable String id) {
        Long userId = SecurityUser.getUserId();
        List<AgentVoicePrintVO> list = agentVoicePrintService.list(userId, id);
        return new Result<List<AgentVoicePrintVO>>().ok(list);
    }



}



