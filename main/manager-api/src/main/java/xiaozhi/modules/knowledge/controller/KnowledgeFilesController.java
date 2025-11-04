package xiaozhi.modules.knowledge.controller;

import java.util.List;
import java.util.Map;

import org.apache.shiro.authz.annotation.RequiresPermissions;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.AllArgsConstructor;
import xiaozhi.common.page.PageData;
import xiaozhi.common.utils.Result;
import xiaozhi.modules.knowledge.dto.KnowledgeFilesDTO;
import xiaozhi.modules.knowledge.service.KnowledgeFilesService;

@AllArgsConstructor
@RestController
@RequestMapping("/api/v1/datasets/{dataset_id}")
@Tag(name = "知识库文档管理")
public class KnowledgeFilesController {

    private final KnowledgeFilesService knowledgeFilesService;

    @GetMapping("/documents")
    @Operation(summary = "分页查询文档列表")
    @RequiresPermissions("sys:role:normal")
    public Result<PageData<KnowledgeFilesDTO>> getPageList(
            @PathVariable("dataset_id") String datasetId,
            @RequestParam(required = false) String name,
            @RequestParam(required = false, defaultValue = "1") Integer page,
            @RequestParam(required = false, defaultValue = "10") Integer page_size) {
        KnowledgeFilesDTO knowledgeFilesDTO = new KnowledgeFilesDTO();
        knowledgeFilesDTO.setDatasetId(datasetId);
        knowledgeFilesDTO.setName(name);
        PageData<KnowledgeFilesDTO> pageData = knowledgeFilesService.getPageList(knowledgeFilesDTO, page, page_size);
        return new Result<PageData<KnowledgeFilesDTO>>().ok(pageData);
    }

    @PostMapping("/documents")
    @Operation(summary = "上传文档到知识库")
    @RequiresPermissions("sys:role:normal")
    public Result<KnowledgeFilesDTO> uploadDocument(
            @PathVariable("dataset_id") String datasetId,
            @RequestParam("file") MultipartFile file,
            @RequestParam(required = false) String name,
            @RequestParam(required = false) String chunkMethod,
            @RequestParam(required = false) String metaFields,
            @RequestParam(required = false) String parserConfig) {
        
        KnowledgeFilesDTO resp = knowledgeFilesService.uploadDocument(datasetId, file, name, 
            metaFields != null ? parseJsonMap(metaFields) : null, 
            chunkMethod, 
            parserConfig != null ? parseJsonMap(parserConfig) : null);
        return new Result<KnowledgeFilesDTO>().ok(resp);
    }

    @DeleteMapping("/documents/{document_id}")
    @Operation(summary = "删除单个文档")
    @Parameter(name = "document_id", description = "文档ID", required = true)
    @RequiresPermissions("sys:role:normal")
    public Result<Void> delete(@PathVariable("dataset_id") String datasetId,
                              @PathVariable("document_id") String documentId) {
        knowledgeFilesService.deleteByDocumentId(documentId, datasetId);
        return new Result<>();
    }
    
    @PostMapping("/chunks")
    @Operation(summary = "批量解析文档（切块）")
    @RequiresPermissions("sys:role:normal")
    public Result<Void> parseDocuments(@PathVariable("dataset_id") String datasetId,
                                      @RequestBody Map<String, List<String>> requestBody) {
        List<String> documentIds = requestBody.get("document_ids");
        if (documentIds == null || documentIds.isEmpty()) {
            return new Result<Void>().error("document_ids参数不能为空");
        }
        
        boolean success = knowledgeFilesService.parseDocuments(datasetId, documentIds);
        if (success) {
            return new Result<Void>();
        } else {
            return new Result<Void>().error("文档解析失败，文档可能正在处理中");
        }
    }
    
    @PostMapping("/documents/{document_id}/parse")
    @Operation(summary = "解析单个文档（切块）")
    @RequiresPermissions("sys:role:normal")
    public Result<Void> parseDocument(@PathVariable("dataset_id") String datasetId,
                                     @PathVariable("document_id") String documentId) {
        List<String> documentIds = java.util.Arrays.asList(documentId);
        
        boolean success = knowledgeFilesService.parseDocuments(datasetId, documentIds);
        if (success) {
            return new Result<Void>();
        } else {
            return new Result<Void>().error("文档解析失败，文档可能正在处理中");
        }
    }

    @PostMapping("/documents/{document_id}/chunks")
    @Operation(summary = "添加切片到指定文档")
    @RequiresPermissions("sys:role:normal")
    public Result<Map<String, Object>> addChunk(@PathVariable("dataset_id") String datasetId,
                                               @PathVariable("document_id") String documentId,
                                               @RequestBody Map<String, Object> requestBody) {
        String content = (String) requestBody.get("content");
        List<String> importantKeywords = (List<String>) requestBody.get("important_keywords");
        List<String> questions = (List<String>) requestBody.get("questions");
        
        Map<String, Object> result = knowledgeFilesService.addChunk(datasetId, documentId, content, importantKeywords, questions);
        return new Result<Map<String, Object>>().ok(result);
    }

    @GetMapping("/documents/{document_id}/chunks")
    @Operation(summary = "列出指定文档的切片")
    @RequiresPermissions("sys:role:normal")
    public Result<Map<String, Object>> listChunks(@PathVariable("dataset_id") String datasetId,
                                                @PathVariable("document_id") String documentId,
                                                @RequestParam(required = false) String keywords,
                                                @RequestParam(required = false, defaultValue = "1") Integer page,
                                                @RequestParam(required = false, defaultValue = "1024") Integer page_size,
                                                @RequestParam(required = false) String id) {
        Map<String, Object> result = knowledgeFilesService.listChunks(datasetId, documentId, keywords, page, page_size, id);
        return new Result<Map<String, Object>>().ok(result);
    }

    /**
     * 召回测试
     */
    @PostMapping("/retrieval-test")
    @Operation(summary = "召回测试")
    @RequiresPermissions("sys:role:normal")
    public Result<Map<String, Object>> retrievalTest(@PathVariable("dataset_id") String datasetId,
                                                   @RequestBody Map<String, Object> params) {
        try {
            // 提取参数
            String question = (String) params.get("question");
            if (question == null || question.trim().isEmpty()) {
                return new Result<Map<String, Object>>().error("问题不能为空");
            }

            List<String> datasetIds = (List<String>) params.get("dataset_ids");
            List<String> documentIds = (List<String>) params.get("document_ids");
            Integer page = (Integer) params.get("page");
            Integer pageSize = (Integer) params.get("page_size");
            Float similarityThreshold = (Float) params.get("similarity_threshold");
            Float vectorSimilarityWeight = (Float) params.get("vector_similarity_weight");
            Integer topK = (Integer) params.get("top_k");
            String rerankId = (String) params.get("rerank_id");
            Boolean keyword = (Boolean) params.get("keyword");
            Boolean highlight = (Boolean) params.get("highlight");
            List<String> crossLanguages = (List<String>) params.get("cross_languages");
            Map<String, Object> metadataCondition = (Map<String, Object>) params.get("metadata_condition");

            // 如果未指定数据集ID，使用当前数据集
            if (datasetIds == null || datasetIds.isEmpty()) {
                datasetIds = java.util.Arrays.asList(datasetId);
            }

            Map<String, Object> result = knowledgeFilesService.retrievalTest(
                question, datasetIds, documentIds, page, pageSize, similarityThreshold,
                vectorSimilarityWeight, topK, rerankId, keyword, highlight, crossLanguages, metadataCondition
            );
            
            return new Result<Map<String, Object>>().ok(result);
        } catch (Exception e) {
            return new Result<Map<String, Object>>().error("召回测试失败: " + e.getMessage());
        }
    }
    
    /**
     * 解析JSON字符串为Map对象
     */
    private Map<String, Object> parseJsonMap(String jsonString) {
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            return objectMapper.readValue(jsonString, new TypeReference<Map<String, Object>>() {});
        } catch (Exception e) {
            throw new RuntimeException("解析JSON字符串失败: " + jsonString, e);
        }
    }
}