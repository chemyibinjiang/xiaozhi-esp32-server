package xiaozhi.modules.knowledge.controller;

import org.apache.shiro.authz.annotation.RequiresPermissions;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.apache.commons.lang3.StringUtils;
import xiaozhi.common.exception.ErrorCode;
import xiaozhi.common.exception.RenException;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.AllArgsConstructor;
import xiaozhi.common.page.PageData;
import xiaozhi.common.utils.Result;
import xiaozhi.modules.knowledge.dto.KnowledgeBaseDTO;
import xiaozhi.modules.knowledge.service.KnowledgeBaseService;
import java.util.Map;


@AllArgsConstructor
@RestController
@RequestMapping("/api/v1")
@Tag(name = "知识库管理")
public class KnowledgeBaseController {

    private final KnowledgeBaseService knowledgeBaseService;

    @GetMapping("/datasets")
    @Operation(summary = "分页查询知识库列表")
    @RequiresPermissions("sys:role:normal")
    public Result<PageData<KnowledgeBaseDTO>> getPageList(
            @RequestParam(required = false) String name,
            @RequestParam(required = false) String id,
            @RequestParam(required = false, defaultValue = "1") Integer page,
            @RequestParam(required = false, defaultValue = "10") Integer page_size,
            @RequestParam(required = false) String orderby,
            @RequestParam(required = false) Boolean desc) {
        KnowledgeBaseDTO knowledgeBaseDTO = new KnowledgeBaseDTO();
        knowledgeBaseDTO.setName(name);
        knowledgeBaseDTO.setDatasetId(id);
        PageData<KnowledgeBaseDTO> pageData = knowledgeBaseService.getPageList(knowledgeBaseDTO, String.valueOf(page), String.valueOf(page_size));
        return new Result<PageData<KnowledgeBaseDTO>>().ok(pageData);
    }

    @GetMapping("/datasets/{dataset_id}")
    @Operation(summary = "根据知识库ID获取知识库详情")
    @RequiresPermissions("sys:role:normal")
    public Result<KnowledgeBaseDTO> getByDatasetId(@PathVariable("dataset_id") String datasetId) {
        KnowledgeBaseDTO knowledgeBaseDTO = knowledgeBaseService.getByDatasetId(datasetId);
        return new Result<KnowledgeBaseDTO>().ok(knowledgeBaseDTO);
    }

    @PostMapping("/datasets")
    @Operation(summary = "创建知识库")
    @RequiresPermissions("sys:role:normal")
    public Result<KnowledgeBaseDTO> save(@RequestBody @Validated KnowledgeBaseDTO knowledgeBaseDTO) {
        KnowledgeBaseDTO resp = knowledgeBaseService.save(knowledgeBaseDTO);
        return new Result<KnowledgeBaseDTO>().ok(resp);
    }

    @PutMapping("/datasets/{dataset_id}")
    @Operation(summary = "更新知识库")
    @RequiresPermissions("sys:role:normal")
    public Result<KnowledgeBaseDTO> update(@PathVariable("dataset_id") String datasetId, 
                                          @RequestBody @Validated KnowledgeBaseDTO knowledgeBaseDTO) {
        knowledgeBaseDTO.setDatasetId(datasetId);
        KnowledgeBaseDTO resp = knowledgeBaseService.update(knowledgeBaseDTO);
        return new Result<KnowledgeBaseDTO>().ok(resp);
    }

    @DeleteMapping("/datasets/{dataset_id}")
    @Operation(summary = "删除单个知识库")
    @Parameter(name = "dataset_id", description = "知识库ID", required = true)
    @RequiresPermissions("sys:role:normal")
    public Result<Void> delete(@PathVariable("dataset_id") String datasetId) {
        knowledgeBaseService.deleteByDatasetId(datasetId);
        return new Result<>();
    }
    
    @DeleteMapping("/datasets/batch")
    @Operation(summary = "批量删除知识库")
    @Parameter(name = "ids", description = "知识库ID列表，用逗号分隔", required = true)
    @RequiresPermissions("sys:role:normal")
    public Result<Void> deleteBatch(@RequestParam("ids") String ids) {
        if (StringUtils.isBlank(ids)) {
            throw new RenException(ErrorCode.PARAMS_GET_ERROR);
        }
        
        String[] idArray = ids.split(",");
        for (String datasetId : idArray) {
            if (StringUtils.isNotBlank(datasetId)) {
                knowledgeBaseService.deleteByDatasetId(datasetId.trim());
            }
        }
        return new Result<>();
    }
    
    @GetMapping("/rag-config/default")
    @Operation(summary = "获取默认RAG配置")
    @RequiresPermissions("sys:role:normal")
    public Result<Map<String, Object>> getDefaultRAGConfig() {
        Map<String, Object> config = knowledgeBaseService.getDefaultRAGConfig();
        return new Result<Map<String, Object>>().ok(config);
    }
}