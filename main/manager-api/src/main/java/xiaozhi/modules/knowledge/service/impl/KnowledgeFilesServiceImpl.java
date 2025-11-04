package xiaozhi.modules.knowledge.service.impl;

import java.io.IOException;
import java.util.*;
import java.io.InputStream;
import java.util.Arrays;
import java.text.SimpleDateFormat; 

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.fasterxml.jackson.databind.ObjectMapper;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.core.io.AbstractResource;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.HttpServerErrorException;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.multipart.MultipartFile;

import xiaozhi.common.constant.Constant;
import xiaozhi.common.exception.ErrorCode;
import xiaozhi.common.exception.RenException;
import xiaozhi.common.page.PageData;
import xiaozhi.modules.knowledge.dto.KnowledgeFilesDTO;
import xiaozhi.modules.knowledge.service.KnowledgeFilesService;
import xiaozhi.modules.model.service.ModelConfigService;
import xiaozhi.modules.model.dao.ModelConfigDao;
import xiaozhi.modules.model.entity.ModelConfigEntity;

@Service
@AllArgsConstructor
@Slf4j
public class KnowledgeFilesServiceImpl implements KnowledgeFilesService {

    private final ModelConfigService modelConfigService;
    private final ModelConfigDao modelConfigDao;
    private RestTemplate restTemplate = new RestTemplate();
    private ObjectMapper objectMapper = new ObjectMapper();

    @Override
    public PageData<KnowledgeFilesDTO> getPageList(KnowledgeFilesDTO knowledgeFilesDTO, Integer page, Integer limit) {
        try {
            
            log.info("=== 开始获取文档列表 ===");
            log.info("查询条件: datasetId={}, name={}, status={}, page={}, limit={}", 
                    knowledgeFilesDTO != null ? knowledgeFilesDTO.getDatasetId() : null,
                    knowledgeFilesDTO != null ? knowledgeFilesDTO.getName() : null,
                    knowledgeFilesDTO != null ? knowledgeFilesDTO.getStatus() : null,
                    page, limit);

            // 获取RAG配置
            Map<String, Object> ragConfig = getRAGConfig();
            String baseUrl = (String) ragConfig.get("base_url");
            String apiKey = (String) ragConfig.get("api_key");

            // 构建请求URL - 根据RAGFlow API文档，获取文档列表的接口
            String datasetId = knowledgeFilesDTO != null ? knowledgeFilesDTO.getDatasetId() : null;
            if (StringUtils.isBlank(datasetId)) {
                throw new RenException(ErrorCode.PARAMS_GET_ERROR, "datasetId不能为空");
            }

            String url = baseUrl + "/api/v1/datasets/" + datasetId + "/documents";
            
            // 添加查询参数
            StringBuilder urlBuilder = new StringBuilder(url);
            List<String> params = new ArrayList<>();
            
            if (knowledgeFilesDTO != null && StringUtils.isNotBlank(knowledgeFilesDTO.getName())) {
                params.add("keywords=" + knowledgeFilesDTO.getName());
            }
            if (page > 0) {
                params.add("page=" + page);
            }
            if (limit > 0) {
                params.add("page_size=" + limit);
            }
            
            if (!params.isEmpty()) {
                urlBuilder.append("?").append(String.join("&", params));
            }
            
            url = urlBuilder.toString();
            log.debug("请求URL: {}", url);

            // 设置请求头
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.set("Authorization", "Bearer " + apiKey);

            HttpEntity<String> requestEntity = new HttpEntity<>(headers);

            // 发送GET请求
            log.info("发送GET请求到RAGFlow API获取文档列表...");
            ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.GET, requestEntity, String.class);

            log.info("RAGFlow API响应状态码: {}", response.getStatusCode());

            if (!response.getStatusCode().is2xxSuccessful()) {
                log.error("RAGFlow API调用失败，状态码: {}, 响应内容: {}", response.getStatusCode(), response.getBody());
                throw new RenException(ErrorCode.RAG_API_QUERY_FAILED);
            }

            String responseBody = response.getBody();
            log.debug("RAGFlow API响应内容: {}", responseBody);

            // 解析响应
            Map<String, Object> responseMap = objectMapper.readValue(responseBody, Map.class);
            Integer code = (Integer) responseMap.get("code");
            
            if (code != null && code == 0) {
                Object dataObj = responseMap.get("data");
                return parseDocumentListResponse(dataObj, page, limit);
            } else {
                log.error("RAGFlow API调用失败，响应码: {}, 响应内容: {}", code, responseBody);
                throw new RenException(ErrorCode.RAG_API_QUERY_FAILED, "RAGFlow API调用失败，响应码: " + code);
            }

        } catch (IOException e) {
            log.error("解析RAGFlow API响应失败: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_QUERY_FAILED, "解析RAGFlow响应失败: " + e.getMessage());
        } catch (HttpClientErrorException e) {
            log.error("RAGFlow API调用失败 - HTTP错误: {}, 状态码: {}, 响应内容: {}", 
                e.getMessage(), e.getStatusCode(), e.getResponseBodyAsString(), e);
            throw new RenException(ErrorCode.RAG_API_QUERY_FAILED, "获取RAGFlow文档列表失败: " + e.getMessage());
        } catch (HttpServerErrorException e) {
            log.error("RAGFlow API调用失败 - 服务器错误: {}, 状态码: {}, 响应内容: {}", 
                e.getMessage(), e.getStatusCode(), e.getResponseBodyAsString(), e);
            throw new RenException(ErrorCode.RAG_API_QUERY_FAILED, "获取RAGFlow文档列表失败: " + e.getMessage());
        } catch (ResourceAccessException e) {
            log.error("RAGFlow API调用失败 - 网络连接错误: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_QUERY_FAILED, "获取RAGFlow文档列表失败: 网络连接错误 - " + e.getMessage());
        } catch (Exception e) {
            log.error("获取文档列表失败: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_QUERY_FAILED, "获取文档列表失败: " + e.getMessage());
        } finally {
            log.info("=== 获取文档列表操作结束 ===");
        }
    }

    /**
     * 解析RAGFlow API返回的文档列表响应
     */
    private PageData<KnowledgeFilesDTO> parseDocumentListResponse(Object dataObj, long curPage, long pageSize) {
        try {
            List<KnowledgeFilesDTO> documents = new ArrayList<>();
            long totalCount = 0;

            if (dataObj instanceof Map) {
                Map<String, Object> dataMap = (Map<String, Object>) dataObj;
                
                // 获取文档列表 - 支持多种可能的字段名
                Object documentsObj = null;
                
                // 支持多种可能的文档列表字段名
                String[] possibleDocumentFields = {"docs", "documents", "items", "list", "data"};
                
                for (String fieldName : possibleDocumentFields) {
                    if (dataMap.containsKey(fieldName) && dataMap.get(fieldName) instanceof List) {
                        documentsObj = dataMap.get(fieldName);
                        log.debug("使用字段名'{}'获取文档列表", fieldName);
                        break;
                    }
                }
                
                // 如果标准字段不存在，尝试自动检测
                if (documentsObj == null) {
                    for (Map.Entry<String, Object> entry : dataMap.entrySet()) {
                        if (entry.getValue() instanceof List) {
                            List<?> list = (List<?>) entry.getValue();
                            if (!list.isEmpty() && list.get(0) instanceof Map) {
                                documentsObj = entry.getValue();
                                log.warn("自动检测到文档列表字段: '{}'，建议检查RAGFlow API文档", entry.getKey());
                                break;
                            }
                        }
                    }
                }
                
                if (documentsObj instanceof List) {
                    List<Map<String, Object>> documentList = (List<Map<String, Object>>) documentsObj;
                    
                    for (Map<String, Object> docMap : documentList) {
                        KnowledgeFilesDTO dto = convertRAGDocumentToDTO(docMap);
                        if (dto != null) {
                            documents.add(dto);
                        }
                    }
                }
                
                // 解析总数 - 支持多种可能的字段名
                Object totalObj = null;
                String[] possibleTotalFields = {"total", "totalCount", "total_count", "count"};
                
                for (String fieldName : possibleTotalFields) {
                    if (dataMap.containsKey(fieldName)) {
                        totalObj = dataMap.get(fieldName);
                        log.debug("使用字段名'{}'获取总数", fieldName);
                        break;
                    }
                }
                
                if (totalObj instanceof Integer) {
                    totalCount = ((Integer) totalObj).longValue();
                } else if (totalObj instanceof Long) {
                    totalCount = (Long) totalObj;
                } else if (totalObj instanceof String) {
                    try {
                        totalCount = Long.parseLong((String) totalObj);
                    } catch (NumberFormatException e) {
                        log.warn("无法解析总数字段: {}", totalObj);
                    }
                }
            }

            // 创建分页数据
            PageData<KnowledgeFilesDTO> pageData = new PageData<>(documents, totalCount);

            log.info("获取文档列表成功，共{}个文档，总数: {}", documents.size(), totalCount);
            return pageData;

        } catch (Exception e) {
            log.error("获取文档列表响应失败: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_QUERY_FAILED, "获取文档列表响应失败: " + e.getMessage());
        }
    }

    /**
     * 将RAGFlow文档数据转换为KnowledgeFilesDTO
     */
    private KnowledgeFilesDTO convertRAGDocumentToDTO(Map<String, Object> docMap) {
        try {
            if (docMap == null) return null;

            KnowledgeFilesDTO dto = new KnowledgeFilesDTO();
            
            // 设置基本字段 - 支持多种可能的字段名
            dto.setId(getStringValueFromMultipleKeys(docMap, "id", "document_id", "doc_id")); // 使用RAGFlow的文档ID作为本地ID
            dto.setDocumentId(getStringValueFromMultipleKeys(docMap, "id", "document_id", "doc_id")); // RAGFlow文档ID
            dto.setName(getStringValueFromMultipleKeys(docMap, "name", "filename", "file_name", "title"));
            dto.setDatasetId(getStringValueFromMultipleKeys(docMap, "dataset_id", "dataset", "knowledge_base_id"));
            
            // 设置文件信息 - 支持多种可能的字段名
            dto.setFileType(getStringValueFromMultipleKeys(docMap, "file_type", "type", "format", "extension"));
            
            // 文件大小 - 支持多种可能的字段名
            Object sizeObj = getValueFromMultipleKeys(docMap, "size", "file_size", "size_bytes");
            if (sizeObj instanceof Integer) {
                dto.setFileSize(((Integer) sizeObj).longValue());
            } else if (sizeObj instanceof Long) {
                dto.setFileSize((Long) sizeObj);
            } else if (sizeObj instanceof String) {
                try {
                    dto.setFileSize(Long.parseLong((String) sizeObj));
                } catch (NumberFormatException e) {
                    log.warn("无法解析文件大小: {}", sizeObj);
                }
            }
            
            // 设置状态 - 支持多种可能的字段名
            Object statusObj = getValueFromMultipleKeys(docMap, "status", "state", "processing_status");
            if (statusObj instanceof String) {
                dto.setStatus(convertRAGStatusToLocal((String) statusObj));
            } else if (statusObj instanceof Integer) {
                dto.setStatus((Integer) statusObj);
            }
            
            // 设置元数据和配置 - 支持多种可能的字段名
            Object metaFieldsObj = getValueFromMultipleKeys(docMap, "meta_fields", "metadata", "meta", "properties");
            if (metaFieldsObj instanceof Map) {
                dto.setMetaFields((Map<String, Object>) metaFieldsObj);
            }
            
            Object parserConfigObj = getValueFromMultipleKeys(docMap, "parser_config", "parser", "parse_config", "config");
            if (parserConfigObj instanceof Map) {
                dto.setParserConfig((Map<String, Object>) parserConfigObj);
            }
            
            dto.setChunkMethod(getStringValueFromMultipleKeys(docMap, "chunk_method", "chunking", "chunk_strategy"));
            
            // 设置时间信息 - 支持多种可能的字段名
            Object createTimeObj = getValueFromMultipleKeys(docMap, "create_time", "created_at", "creation_time", "created");
            if (createTimeObj instanceof Long) {
                dto.setCreatedAt(new Date((Long) createTimeObj));
            } else if (createTimeObj instanceof String) {
                // 尝试解析时间字符串
                try {
                    // 这里可以根据实际的时间格式进行调整
                    dto.setCreatedAt(new Date(Long.parseLong((String) createTimeObj)));
                } catch (NumberFormatException e) {
                    log.warn("无法解析创建时间: {}", createTimeObj);
                }
            }
            
            Object updateTimeObj = getValueFromMultipleKeys(docMap, "update_time", "updated_at", "modified_time", "modified");
            if (updateTimeObj instanceof Long) {
                dto.setUpdatedAt(new Date((Long) updateTimeObj));
            } else if (updateTimeObj instanceof String) {
                try {
                    dto.setUpdatedAt(new Date(Long.parseLong((String) updateTimeObj)));
                } catch (NumberFormatException e) {
                    log.warn("无法解析更新时间: {}", updateTimeObj);
                }
            }

            return dto;
            
        } catch (Exception e) {
            log.error("转换RAGFlow文档数据失败: {}", e.getMessage(), e);
            return null;
        }
    }

    /**
     * 转换RAGFlow状态到本地状态
     */
    private Integer convertRAGStatusToLocal(String ragStatus) {
        if (ragStatus == null) return 0;
        
        switch (ragStatus.toLowerCase()) {
            case "pending":
            case "processing":
                return 1; // 处理中
            case "completed":
            case "finished":
                return 2; // 已完成
            case "failed":
            case "error":
                return 3; // 失败
            default:
                return 0; // 未知
        }
    }

    /**
     * 从Map中获取字符串值
     */
    private String getStringValue(Map<String, Object> map, String key) {
        Object value = map.get(key);
        return value != null ? value.toString() : null;
    }

    /**
     * 从多个可能的字段名中获取字符串值
     */
    private String getStringValueFromMultipleKeys(Map<String, Object> map, String... keys) {
        for (String key : keys) {
            Object value = map.get(key);
            if (value != null) {
                return value.toString();
            }
        }
        return null;
    }

    /**
     * 从多个可能的字段名中获取值
     */
    private Object getValueFromMultipleKeys(Map<String, Object> map, String... keys) {
        for (String key : keys) {
            Object value = map.get(key);
            if (value != null) {
                return value;
            }
        }
        return null;
    }

    @Override
    public KnowledgeFilesDTO getById(String id) {
        if (StringUtils.isBlank(id)) {
            throw new RenException(ErrorCode.IDENTIFIER_NOT_NULL);
        }

        log.info("=== 开始根据ID获取文档 ===");
        log.info("文档ID: {}", id);

        try {
            KnowledgeFilesDTO queryDTO = new KnowledgeFilesDTO();
            
            throw new RenException(ErrorCode.PARAMS_GET_ERROR, "请使用getByDocumentId方法，并提供datasetId");

        } catch (Exception e) {
            log.error("根据ID获取文档失败: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_QUERY_FAILED, "获取文档失败: " + e.getMessage());
        } finally {
            log.info("=== 根据ID获取文档操作结束 ===");
        }
    }

    @Override
    public KnowledgeFilesDTO getByDocumentId(String documentId) {
        // 重载方法，保持向后兼容，但实际需要datasetId
        throw new RenException(ErrorCode.PARAMS_GET_ERROR, "请使用getByDocumentId(documentId, datasetId)方法，并提供datasetId");
    }

    @Override
    public KnowledgeFilesDTO getByDocumentId(String documentId, String datasetId) {
        if (StringUtils.isBlank(documentId) || StringUtils.isBlank(datasetId)) {
            throw new RenException(ErrorCode.PARAMS_GET_ERROR, "documentId和datasetId不能为空");
        }

        log.info("=== 开始根据documentId获取文档 ===");
        log.info("documentId: {}, datasetId: {}", documentId, datasetId);

        try {
            // 获取RAG配置
            Map<String, Object> ragConfig = getRAGConfig();
            String baseUrl = (String) ragConfig.get("base_url");
            String apiKey = (String) ragConfig.get("api_key");

            // 修正API路径 - 根据RAGFlow API规范，获取单个文档需要datasetId
            String url = baseUrl + "/api/v1/datasets/" + datasetId + "/documents/" + documentId;
            log.debug("请求URL: {}", url);

            // 设置请求头
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.set("Authorization", "Bearer " + apiKey);

            HttpEntity<String> requestEntity = new HttpEntity<>(headers);

            // 发送GET请求
            log.info("发送GET请求到RAGFlow API获取文档详情...");
            ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.GET, requestEntity, String.class);

            log.info("RAGFlow API响应状态码: {}", response.getStatusCode());

            if (!response.getStatusCode().is2xxSuccessful()) {
                log.error("RAGFlow API调用失败，状态码: {}, 响应内容: {}", response.getStatusCode(), response.getBody());
                throw new RenException(ErrorCode.RAG_API_QUERY_FAILED);
            }

            String responseBody = response.getBody();
            log.debug("RAGFlow API响应内容: {}", responseBody);

            // 解析响应
            Map<String, Object> responseMap = objectMapper.readValue(responseBody, Map.class);
            Integer code = (Integer) responseMap.get("code");
            
            if (code != null && code == 0) {
                Object dataObj = responseMap.get("data");
                if (dataObj instanceof Map) {
                    KnowledgeFilesDTO dto = convertRAGDocumentToDTO((Map<String, Object>) dataObj);
                    if (dto != null) {
                        log.info("获取文档详情成功，documentId: {}", documentId);
                        return dto;
                    }
                }
                throw new RenException(ErrorCode.Knowledge_Base_RECORD_NOT_EXISTS);
            } else {
                log.error("RAGFlow API调用失败，响应码: {}, 响应内容: {}", code, responseBody);
                throw new RenException(ErrorCode.RAG_API_QUERY_FAILED, "RAGFlow API调用失败，响应码: " + code);
            }

        } catch (IOException e) {
            log.error("解析RAGFlow API响应失败: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_QUERY_FAILED, "解析RAGFlow响应失败: " + e.getMessage());
        } catch (HttpClientErrorException e) {
            if (e.getStatusCode() == HttpStatus.NOT_FOUND) {
                log.warn("文档不存在，documentId: {}, datasetId: {}", documentId, datasetId);
                throw new RenException(ErrorCode.Knowledge_Base_RECORD_NOT_EXISTS);
            }
            log.error("RAGFlow API调用失败 - HTTP错误: {}, 状态码: {}, 响应内容: {}", 
                e.getMessage(), e.getStatusCode(), e.getResponseBodyAsString(), e);
            throw new RenException(ErrorCode.RAG_API_QUERY_FAILED, "获取RAGFlow文档失败: " + e.getMessage());
        } catch (HttpServerErrorException e) {
            log.error("RAGFlow API调用失败 - 服务器错误: {}, 状态码: {}, 响应内容: {}", 
                e.getMessage(), e.getStatusCode(), e.getResponseBodyAsString(), e);
            throw new RenException(ErrorCode.RAG_API_QUERY_FAILED, "获取RAGFlow文档失败: " + e.getMessage());
        } catch (ResourceAccessException e) {
            log.error("RAGFlow API调用失败 - 网络连接错误: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_QUERY_FAILED, "获取RAGFlow文档失败: 网络连接错误 - " + e.getMessage());
        } catch (Exception e) {
            log.error("根据documentId获取文档失败: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_QUERY_FAILED, "获取文档失败: " + e.getMessage());
        } finally {
            log.info("=== 根据documentId获取文档操作结束 ===");
        }
    }

    @Override
    public KnowledgeFilesDTO uploadDocument(String datasetId, MultipartFile file, String name, 
                                           Map<String, Object> metaFields, String chunkMethod, 
                                           Map<String, Object> parserConfig) {
        if (StringUtils.isBlank(datasetId) || file == null || file.isEmpty()) {
            throw new RenException(ErrorCode.PARAMS_GET_ERROR);
        }

        log.info("=== 开始文档上传操作 ===");
        log.info("上传文档到数据集: {}, 文件名: {}", datasetId, file.getOriginalFilename());

        try {
            // 在文件上传前添加详细日志
            log.info("1. 开始处理文件信息");
            String fileName = file.getOriginalFilename();
            String fileType = getFileType(fileName);
            long fileSize = file.getSize();
            
            log.info("文件信息 - 文件名: {}, 文件类型: {}, 文件大小: {} bytes", 
                    fileName, fileType, fileSize);
            
            // 检查文件基本信息
            if (StringUtils.isBlank(fileName)) {
                log.error("文件名为空");
                throw new RenException(ErrorCode.PARAMS_GET_ERROR, "文件名不能为空");
            }
            
            if (fileSize == 0) {
                log.error("文件大小为0");
                throw new RenException(ErrorCode.PARAMS_GET_ERROR, "文件内容为空");
            }

            log.info("2. 开始流式上传到RAGFlow");
            // 直接调用RAGFlow API上传文档 - 使用流式上传
            String documentId = uploadDocumentToRAGFlow(datasetId, file, name, metaFields, chunkMethod, parserConfig);
            
            log.info("文档上传成功，documentId: {}", documentId);

            // 返回上传的文档信息
            KnowledgeFilesDTO result = new KnowledgeFilesDTO();
            result.setId(documentId); // 使用documentId作为ID
            result.setDocumentId(documentId);
            result.setDatasetId(datasetId);
            result.setName(StringUtils.isNotBlank(name) ? name : fileName);
            result.setFileType(fileType);
            result.setFileSize(fileSize);
            result.setStatus(1); // 上传成功，设置为处理中状态

            return result;
            
        } catch (Exception e) {
            log.error("文档上传失败: {}", e.getMessage());
            log.error("文档上传失败详细异常: ", e);
            if (e instanceof RenException) {
                throw (RenException) e;
            }
            throw new RenException(ErrorCode.INTERNAL_SERVER_ERROR);
        } finally {
            log.info("=== 文档上传操作结束 ===");
        }
    }

    @Override
    public KnowledgeFilesDTO update(KnowledgeFilesDTO knowledgeFilesDTO) {
        if (knowledgeFilesDTO == null || StringUtils.isBlank(knowledgeFilesDTO.getDocumentId())) {
            throw new RenException(ErrorCode.IDENTIFIER_NOT_NULL);
        }

        log.info("=== 开始更新文档操作 ===");
        log.info("更新文档documentId: {}", knowledgeFilesDTO.getDocumentId());

        try {
            // 调用RAGFlow API更新文档配置
            log.info("开始调用RAGFlow API更新文档配置");
            updateDocumentInRAGFlow(
                knowledgeFilesDTO.getDocumentId(),
                knowledgeFilesDTO.getName(),
                knowledgeFilesDTO.getMetaFields(),
                knowledgeFilesDTO.getChunkMethod(),
                knowledgeFilesDTO.getParserConfig()
            );
            log.info("RAGFlow API更新调用完成");

            // 返回更新后的文档信息（通过查询获取最新状态）
            // 需要datasetId，这里假设knowledgeFilesDTO中包含datasetId
            if (StringUtils.isBlank(knowledgeFilesDTO.getDatasetId())) {
                throw new RenException(ErrorCode.PARAMS_GET_ERROR, "更新文档需要datasetId");
            }
            return getByDocumentId(knowledgeFilesDTO.getDocumentId(), knowledgeFilesDTO.getDatasetId());

        } catch (Exception e) {
            log.error("更新文档失败: {}", e.getMessage(), e);
            if (e instanceof RenException) {
                throw (RenException) e;
            }
            throw new RenException(ErrorCode.RAG_API_UPDATE_FAILED, "更新文档失败: " + e.getMessage());
        } finally {
            log.info("=== 更新文档操作结束 ===");
        }
    }

    @Override
    public void delete(String documentId) {
        // 重载方法，保持向后兼容，但实际需要datasetId
        throw new RenException(ErrorCode.PARAMS_GET_ERROR, "请使用deleteByDocumentId(documentId, datasetId)方法，并提供datasetId");
    }

    @Override
    public void deleteByDocumentId(String documentId) {
        // 重载方法，保持向后兼容，但实际需要datasetId
        throw new RenException(ErrorCode.PARAMS_GET_ERROR, "请使用deleteByDocumentId(documentId, datasetId)方法，并提供datasetId");
    }

    @Override
    public void deleteByDocumentId(String documentId, String datasetId) {
        if (StringUtils.isBlank(documentId) || StringUtils.isBlank(datasetId)) {
            throw new RenException(ErrorCode.PARAMS_GET_ERROR, "documentId和datasetId不能为空");
        }

        log.info("=== 开始根据documentId删除文档 ===");
        log.info("删除文档documentId: {}, datasetId: {}", documentId, datasetId);

        try {
            // 直接调用RAGFlow API删除文档，不进行前置验证
            // 因为即使文档不存在，RAGFlow API也会返回相应的错误信息
            log.info("开始调用RAGFlow API删除文档");
            deleteDocumentInRAGFlow(documentId, datasetId);
            log.info("RAGFlow API删除调用完成");

        } catch (Exception e) {
            log.error("删除文档失败: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_DELETE_FAILED, "删除文档失败: " + e.getMessage());
        } finally {
            log.info("=== 根据documentId删除文档操作结束 ===");
        }
    }

    @Override
    public void deleteBatch(List<String> ids) {
        if (ids == null || ids.isEmpty()) {
            throw new RenException(ErrorCode.PARAMS_GET_ERROR);
        }

        log.info("=== 开始批量删除文档操作 ===");
        log.info("批量删除文档数量: {}", ids.size());

        // 由于批量删除需要datasetId，这里无法处理
        throw new RenException(ErrorCode.PARAMS_GET_ERROR, "批量删除需要datasetId，请使用单个删除接口");
    }

    

    /**
     * 获取文件类型 - 支持RAGFlow四种文档格式类型
     */
    private String getFileType(String fileName) {
        if (StringUtils.isBlank(fileName)) {
            log.warn("文件名为空，返回unknown类型");
            return "unknown";
        }
        
        try {
            int lastDotIndex = fileName.lastIndexOf('.');
            if (lastDotIndex > 0 && lastDotIndex < fileName.length() - 1) {
                String extension = fileName.substring(lastDotIndex + 1).toLowerCase();
                
                // 支持RAGFlow四种文档格式类型
                String[] documentTypes = {"pdf", "doc", "docx", "txt", "md", "mdx"};
                String[] spreadsheetTypes = {"csv", "xls", "xlsx"};
                String[] imageTypes = {"jpeg", "jpg", "png", "tif", "gif"};
                String[] presentationTypes = {"ppt", "pptx"};
                
                // 检查文档类型
                for (String type : documentTypes) {
                    if (type.equals(extension)) {
                        return "document";
                    }
                }
                
                // 检查表格类型
                for (String type : spreadsheetTypes) {
                    if (type.equals(extension)) {
                        return "spreadsheet";
                    }
                }
                
                // 检查图片类型
                for (String type : imageTypes) {
                    if (type.equals(extension)) {
                        return "image";
                    }
                }
                
                // 检查幻灯片类型
                for (String type : presentationTypes) {
                    if (type.equals(extension)) {
                        return "presentation";
                    }
                }
                
                // 返回原始扩展名作为文件类型
                return extension;
            }
            return "unknown";
        } catch (Exception e) {
            log.error("获取文件类型失败: ", e);
            return "unknown";
        }
    }

    @Override
    public Map<String, Object> getRAGConfig(String ragModelId) {
        if (StringUtils.isBlank(ragModelId)) {
            throw new RenException(ErrorCode.PARAMS_GET_ERROR);
        }
        
        // 从缓存获取模型配置
        ModelConfigEntity modelConfig = modelConfigService.getModelByIdFromCache(ragModelId);
        if (modelConfig == null || modelConfig.getConfigJson() == null) {
            throw new RenException(ErrorCode.RAG_CONFIG_NOT_FOUND);
        }
        
        // 验证是否为RAG类型配置
        if (!Constant.RAG_CONFIG_TYPE.equals(modelConfig.getModelType().toUpperCase())) {
            throw new RenException(ErrorCode.RAG_CONFIG_TYPE_ERROR);
        }
        
        Map<String, Object> config = modelConfig.getConfigJson();
        
        // 验证必要的配置参数
        validateRagConfig(config);
        
        // 返回配置信息
        return config;
    }
    
    @Override
    public Map<String, Object> getDefaultRAGConfig() {
        // 获取默认RAG模型配置
        QueryWrapper<ModelConfigEntity> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("model_type", Constant.RAG_CONFIG_TYPE)
                   .eq("is_default", 1)
                   .eq("is_enabled", 1);

        List<ModelConfigEntity> modelConfigs = modelConfigDao.selectList(queryWrapper);
        if (modelConfigs == null || modelConfigs.isEmpty()) {
            throw new RenException(ErrorCode.RAG_DEFAULT_CONFIG_NOT_FOUND);
        }
        
        ModelConfigEntity defaultConfig = modelConfigs.get(0);
        if (defaultConfig.getConfigJson() == null) {
            throw new RenException(ErrorCode.RAG_CONFIG_NOT_FOUND);
        }
        
        Map<String, Object> config = defaultConfig.getConfigJson();
        
        // 验证必要的配置参数
        validateRagConfig(config);
        
        return config;
    }
    
    /**
     * 验证RAG配置中是否包含必要的参数
     */
    private void validateRagConfig(Map<String, Object> config) {
        if (config == null) {
            throw new RenException(ErrorCode.RAG_CONFIG_NOT_FOUND);
        }
        
        // 从配置中提取必要的参数
        String baseUrl = (String) config.get("base_url");
        String apiKey = (String) config.get("api_key");
        
        // 验证base_url是否存在且非空
        if (StringUtils.isBlank(baseUrl)) {
            throw new RenException(ErrorCode.RAG_CONFIG_MISSING_PARAMS);
        }
    }

    /**
     * 获取默认RAG配置
     */
    private Map<String, Object> getRAGConfig() {
        return getDefaultRAGConfig();
    }

    /**
     * 调用RAGFlow API上传文档 - 流式上传版本
     */
    private String uploadDocumentToRAGFlow(String datasetId, MultipartFile file, String name, 
                                          Map<String, Object> metaFields, String chunkMethod, 
                                          Map<String, Object> parserConfig) {
        try {
            // 获取RAG配置
            Map<String, Object> ragConfig = getRAGConfig();
            String baseUrl = (String) ragConfig.get("base_url");
            String apiKey = (String) ragConfig.get("api_key");
            
            log.info("开始调用RAGFlow API流式上传文档，datasetId: {}, 文件名: {}", datasetId, file.getOriginalFilename());
            log.debug("RAGFlow配置 - baseUrl: {}, apiKey: {}", baseUrl, StringUtils.isBlank(apiKey) ? "未配置" : "已配置");
            
            // 构建请求URL
            String url = baseUrl + "/api/v1/datasets/" + datasetId + "/documents";
            log.debug("请求URL: {}", url);
            
            // 构建multipart/form-data请求
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);
            headers.set("Authorization", "Bearer " + apiKey);
            
            // 创建多部分请求体 - 使用MultipartFileResource进行流式上传
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", new MultipartFileResource(file, file.getOriginalFilename()));
            
            // 添加其他参数
            if (StringUtils.isNotBlank(name)) {
                body.add("name", name);
            } else {
                body.add("name", file.getOriginalFilename());
            }
            
            if (metaFields != null && !metaFields.isEmpty()) {
                try {
                    body.add("meta_fields", objectMapper.writeValueAsString(metaFields));
                } catch (Exception e) {
                    log.warn("序列化meta_fields失败: {}", e.getMessage());
                }
            }
            
            if (StringUtils.isNotBlank(chunkMethod)) {
                body.add("chunk_method", chunkMethod);
            }
            
            if (parserConfig != null && !parserConfig.isEmpty()) {
                try {
                    body.add("parser_config", objectMapper.writeValueAsString(parserConfig));
                } catch (Exception e) {
                    log.warn("序列化parser_config失败: {}", e.getMessage());
                }
            }
            
            log.debug("multipart请求体参数数量: {}", body.size());
            log.debug("multipart请求体参数: {}", body.keySet());
            
            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);
            
            // 发送POST请求
            log.info("发送multipart/form-data POST请求到RAGFlow API...");
            ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.POST, requestEntity, String.class);
            
            log.info("RAGFlow API响应状态码: {}", response.getStatusCode());
            log.debug("RAGFlow API响应内容: {}", response.getBody());
            
            String responseBody = response.getBody();
            String documentId = null;
            
            if (response.getStatusCode().is2xxSuccessful()) {
                try {
                    Map<String, Object> responseMap = objectMapper.readValue(responseBody, Map.class);
                    
                    log.debug("RAGFlow API响应解析结果: {}", responseMap);
                    
                    // 首先检查响应码
                    Integer code = (Integer) responseMap.get("code");
                    if (code != null && code == 0) {
                        // 响应码为0表示成功，从data字段中获取documentId
                        Object dataObj = responseMap.get("data");
                        
                        // 增强的documentId提取逻辑
                        documentId = extractDocumentIdFromResponse(dataObj);
                        
                        // 如果从data字段无法提取，尝试从根级别提取
                        if (StringUtils.isBlank(documentId)) {
                            documentId = extractDocumentIdFromRoot(responseMap);
                        }
                        
                        log.info("文档上传成功，documentId: {}", documentId);
                    } else {
                        // 如果响应码不为0，说明API调用失败
                        log.error("RAGFlow API调用失败，响应码: {}, 响应内容: {}", code, responseBody);
                        throw new RenException(ErrorCode.RAG_API_CREATE_FAILED, "RAGFlow API调用失败，响应码: " + code);
                    }
                    
                    log.info("从RAGFlow API响应中解析出documentId: {}", documentId);
                    log.debug("完整响应内容: {}", responseBody);
                } catch (Exception e) {
                    log.error("解析RAGFlow API响应失败: {}, 响应内容: {}", e.getMessage(), responseBody);
                    throw new RenException(ErrorCode.RAG_API_CREATE_FAILED, "解析RAGFlow响应失败: " + e.getMessage());
                }
            } else {
                log.error("RAGFlow API调用失败，状态码: {}, 响应内容: {}", response.getStatusCode(), responseBody);
                throw new RenException(ErrorCode.RAG_API_CREATE_FAILED);
            }
            
            if (StringUtils.isBlank(documentId)) {
                log.error("无法从RAGFlow API响应中获取documentId，响应内容: {}", responseBody);
                throw new RenException(ErrorCode.RAG_API_CREATE_FAILED, "RAGFlow API响应中未包含documentId");
            }
            
            log.info("RAGFlow文档上传成功，documentId: {}，文档已开始自动解析切片", documentId);
            return documentId;
            
        } catch (HttpClientErrorException e) {
            log.error("RAGFlow API调用失败 - HTTP错误: {}, 状态码: {}, 响应内容: {}", 
                e.getMessage(), e.getStatusCode(), e.getResponseBodyAsString(), e);
            throw new RenException(ErrorCode.RAG_API_CREATE_FAILED, "上传RAGFlow文档失败: " + e.getMessage() + ", 响应: " + e.getResponseBodyAsString());
        } catch (HttpServerErrorException e) {
            log.error("RAGFlow API调用失败 - 服务器错误: {}, 状态码: {}, 响应内容: {}", 
                e.getMessage(), e.getStatusCode(), e.getResponseBodyAsString(), e);
            throw new RenException(ErrorCode.RAG_API_CREATE_FAILED, "上传RAGFlow文档失败: " + e.getMessage() + ", 响应: " + e.getResponseBodyAsString());
        } catch (ResourceAccessException e) {
            log.error("RAGFlow API调用失败 - 网络连接错误: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_CREATE_FAILED, "上传RAGFlow文档失败: 网络连接错误 - " + e.getMessage());
        } catch (Exception e) {
            log.error("RAGFlow API调用失败 - 未知错误: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_CREATE_FAILED, "上传RAGFlow文档失败: " + e.getMessage());
        }
    }
    
    /**
     * 从响应数据中提取documentId
     */
    private String extractDocumentIdFromResponse(Object dataObj) {
        String documentId = null;
        
        if (dataObj instanceof List) {
            // data是一个数组，取第一个元素的id字段
            List<Map<String, Object>> dataList = (List<Map<String, Object>>) dataObj;
            if (!dataList.isEmpty()) {
                Map<String, Object> firstItem = dataList.get(0);
                documentId = extractDocumentIdFromMap(firstItem);
            }
        } else if (dataObj instanceof Map) {
            // data是一个对象
            Map<String, Object> dataMap = (Map<String, Object>) dataObj;
            documentId = extractDocumentIdFromMap(dataMap);
        }
        
        return documentId;
    }
    
    /**
     * 从Map中提取documentId，支持多种可能的字段名
     */
    private String extractDocumentIdFromMap(Map<String, Object> map) {
        if (map == null) return null;
        
        // 尝试多种可能的字段名
        String[] possibleFieldNames = {"id", "document_id", "documentId", "doc_id", "documentId"};
        
        for (String fieldName : possibleFieldNames) {
            Object value = map.get(fieldName);
            if (value != null && value instanceof String && StringUtils.isNotBlank((String) value)) {
                return (String) value;
            }
        }
        
        return null;
    }
    
    /**
     * 从根级别响应中提取documentId
     */
    private String extractDocumentIdFromRoot(Map<String, Object> responseMap) {
        if (responseMap == null) return null;
        
        // 尝试从根级别提取
        String[] possibleFieldNames = {"id", "document_id", "documentId", "doc_id", "documentId"};
        
        for (String fieldName : possibleFieldNames) {
            Object value = responseMap.get(fieldName);
            if (value != null && value instanceof String && StringUtils.isNotBlank((String) value)) {
                return (String) value;
            }
        }
        
        return null;
    }

    /**
     * 调用RAGFlow API更新文档配置
     */
    private void updateDocumentInRAGFlow(String documentId, String name, Map<String, Object> metaFields, 
                                         String chunkMethod, Map<String, Object> parserConfig) {
        try {
            // 获取RAG配置
            Map<String, Object> ragConfig = getRAGConfig();
            String baseUrl = (String) ragConfig.get("base_url");
            String apiKey = (String) ragConfig.get("api_key");
            
            log.info("开始调用RAGFlow API更新文档配置，documentId: {}, name: {}", documentId, name);
            log.debug("RAGFlow配置 - baseUrl: {}, apiKey: {}", baseUrl, StringUtils.isBlank(apiKey) ? "未配置" : "已配置");
            
            // 构建请求URL - 根据RAGFlow API规范，更新文档需要datasetId，但这里假设documentId足够
            // 如果更新失败，可能需要调整URL格式
            String url = baseUrl + "/api/v1/documents/" + documentId;
            log.debug("请求URL: {}", url);
            
            // 构建请求体
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("document_id", documentId);
            if (StringUtils.isNotBlank(name)) {
                requestBody.put("name", name);
            }
            if (metaFields != null && !metaFields.isEmpty()) {
                requestBody.put("meta_fields", metaFields);
            }
            if (StringUtils.isNotBlank(chunkMethod)) {
                requestBody.put("chunk_method", chunkMethod);
            }
            if (parserConfig != null && !parserConfig.isEmpty()) {
                requestBody.put("parser_config", parserConfig);
            }
            
            log.debug("请求体: {}", requestBody);
            
            // 设置请求头
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.set("Authorization", "Bearer " + apiKey);
            
            HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(requestBody, headers);
            
            // 发送PUT请求
            log.info("发送PUT请求到RAGFlow API...");
            ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.PUT, requestEntity, String.class);
            
            log.info("RAGFlow API响应状态码: {}", response.getStatusCode());
            log.debug("RAGFlow API响应内容: {}", response.getBody());
            
            if (!response.getStatusCode().is2xxSuccessful()) {
                log.error("RAGFlow API调用失败，状态码: {}, 响应内容: {}", response.getStatusCode(), response.getBody());
                throw new RenException(ErrorCode.RAG_API_UPDATE_FAILED);
            }
            
            log.info("RAGFlow文档配置更新成功，documentId: {}", documentId);
            
        } catch (HttpClientErrorException e) {
            log.error("RAGFlow API调用失败 - HTTP错误: {}, 状态码: {}, 响应内容: {}", 
                e.getMessage(), e.getStatusCode(), e.getResponseBodyAsString(), e);
            throw new RenException(ErrorCode.RAG_API_UPDATE_FAILED, "更新RAGFlow文档配置失败: " + e.getMessage() + ", 响应: " + e.getResponseBodyAsString());
        } catch (HttpServerErrorException e) {
            log.error("RAGFlow API调用失败 - 服务器错误: {}, 状态码: {}, 响应内容: {}", 
                e.getMessage(), e.getStatusCode(), e.getResponseBodyAsString(), e);
            throw new RenException(ErrorCode.RAG_API_UPDATE_FAILED, "更新RAGFlow文档配置失败: " + e.getMessage() + ", 响应: " + e.getResponseBodyAsString());
        } catch (ResourceAccessException e) {
            log.error("RAGFlow API调用失败 - 网络连接错误: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_UPDATE_FAILED, "更新RAGFlow文档配置失败: 网络连接错误 - " + e.getMessage());
        } catch (Exception e) {
            log.error("RAGFlow API调用失败 - 未知错误: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_UPDATE_FAILED, "更新RAGFlow文档配置失败: " + e.getMessage());
        }
    }

    /**
     * 调用RAGFlow API删除文档
     */
    private void deleteDocumentInRAGFlow(String documentId, String datasetId) {
        try {
            // 获取RAG配置
            Map<String, Object> ragConfig = getRAGConfig();
            String baseUrl = (String) ragConfig.get("base_url");
            String apiKey = (String) ragConfig.get("api_key");
            
            log.info("开始调用RAGFlow API删除文档，documentId: {}, datasetId: {}", documentId, datasetId);
            log.debug("RAGFlow配置 - baseUrl: {}, apiKey: {}", baseUrl, StringUtils.isBlank(apiKey) ? "未配置" : "已配置");
            
            // 构建请求URL - 根据RAGFlow API文档，使用正确的路径参数名称
            String url = baseUrl + "/api/v1/datasets/" + datasetId + "/documents";
            log.debug("请求URL: {}", url);
            
            // 构建请求体 - 严格按照API文档格式
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("ids", Arrays.asList(documentId)); // 使用Arrays.asList确保序列化正确
            log.debug("请求体: {}", requestBody);
            
            // 设置请求头
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.set("Authorization", "Bearer " + apiKey);
            
            HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(requestBody, headers);
            
            // 发送DELETE请求
            log.info("发送DELETE请求到RAGFlow API...");
            ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.DELETE, requestEntity, String.class);
            
            log.info("RAGFlow API响应状态码: {}", response.getStatusCode());
            log.debug("RAGFlow API响应内容: {}", response.getBody());
            
            String responseBody = response.getBody();
            
            if (response.getStatusCode().is2xxSuccessful()) {
                // 验证响应格式
                if (responseBody != null) {
                    try {
                        Map<String, Object> responseMap = objectMapper.readValue(responseBody, Map.class);
                        Integer code = (Integer) responseMap.get("code");
                        
                        if (code != null && code == 0) {
                            log.info("RAGFlow文档删除成功，documentId: {}", documentId);
                            return;
                        } else {
                            String message = (String) responseMap.get("message");
                            log.error("RAGFlow API调用失败，响应码: {}, 消息: {}", code, message);
                            throw new RenException(ErrorCode.RAG_API_DELETE_FAILED, "RAGFlow API调用失败: " + message);
                        }
                    } catch (Exception e) {
                        log.warn("解析RAGFlow响应失败，但HTTP状态码成功，视为删除成功: {}", e.getMessage());
                        log.info("RAGFlow文档删除成功，documentId: {}", documentId);
                        return;
                    }
                } else {
                    log.info("RAGFlow文档删除成功（无响应体），documentId: {}", documentId);
                    return;
                }
            } else {
                log.error("RAGFlow API调用失败，状态码: {}, 响应内容: {}", response.getStatusCode(), responseBody);
                throw new RenException(ErrorCode.RAG_API_DELETE_FAILED);
            }
            
        } catch (HttpClientErrorException e) {
            log.error("RAGFlow API调用失败 - HTTP错误: {}, 状态码: {}, 响应内容: {}", 
                e.getMessage(), e.getStatusCode(), e.getResponseBodyAsString(), e);
            throw new RenException(ErrorCode.RAG_API_DELETE_FAILED, "删除RAGFlow文档失败: " + e.getMessage() + ", 响应: " + e.getResponseBodyAsString());
        } catch (HttpServerErrorException e) {
            log.error("RAGFlow API调用失败 - 服务器错误: {}, 状态码: {}, 响应内容: {}", 
                e.getMessage(), e.getStatusCode(), e.getResponseBodyAsString(), e);
            throw new RenException(ErrorCode.RAG_API_DELETE_FAILED, "删除RAGFlow文档失败: " + e.getMessage() + ", 响应: " + e.getResponseBodyAsString());
        } catch (ResourceAccessException e) {
            log.error("RAGFlow API调用失败 - 网络连接错误: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_DELETE_FAILED, "删除RAGFlow文档失败: 网络连接错误 - " + e.getMessage());
        } catch (Exception e) {
            log.error("RAGFlow API调用失败 - 未知错误: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_DELETE_FAILED, "删除RAGFlow文档失败: " + e.getMessage());
        }
    }

    /**
     * 辅助类：将MultipartFile转换为Resource用于流式上传
     */
    private static class MultipartFileResource extends AbstractResource {
        private final MultipartFile multipartFile;
        private final String filename;
        
        public MultipartFileResource(MultipartFile multipartFile, String filename) {
            this.multipartFile = multipartFile;
            this.filename = filename;
        }
        
        @Override
        public String getFilename() {
            return this.filename;
        }
        
        @Override
        public InputStream getInputStream() throws IOException {
            return this.multipartFile.getInputStream();
        }
        
        @Override
        public long contentLength() throws IOException {
            return this.multipartFile.getSize();
        }
        
        @Override
        public boolean exists() {
            return true;
        }
        
        @Override
        public String getDescription() {
            return "MultipartFile resource for " + this.filename;
        }
    }

    @Override
    public boolean parseDocuments(String datasetId, List<String> documentIds) {
        if (StringUtils.isBlank(datasetId) || documentIds == null || documentIds.isEmpty()) {
            throw new RenException(ErrorCode.PARAMS_GET_ERROR, "datasetId和documentIds不能为空");
        }

        log.info("=== 开始解析文档（切块） ===");
        log.info("datasetId: {}, documentIds: {}", datasetId, documentIds);

        try {
            // 获取RAG配置
            Map<String, Object> ragConfig = getRAGConfig();
            String baseUrl = (String) ragConfig.get("base_url");
            String apiKey = (String) ragConfig.get("api_key");

            // 构建请求URL - 根据RAGFlow API文档，解析文档的接口
            String url = baseUrl + "/api/v1/datasets/" + datasetId + "/chunks";
            log.debug("请求URL: {}", url);

            // 构建请求体
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("document_ids", documentIds);
            log.debug("请求体: {}", requestBody);

            // 设置请求头
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.set("Authorization", "Bearer " + apiKey);

            HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(requestBody, headers);

            // 发送POST请求
            log.info("发送POST请求到RAGFlow API解析文档...");
            ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.POST, requestEntity, String.class);

            log.info("RAGFlow API响应状态码: {}", response.getStatusCode());

            if (!response.getStatusCode().is2xxSuccessful()) {
                log.error("RAGFlow API调用失败，状态码: {}, 响应内容: {}", response.getStatusCode(), response.getBody());
                throw new RenException(ErrorCode.RAG_API_PARSE_FAILED);
            }

            String responseBody = response.getBody();
            log.debug("RAGFlow API响应内容: {}", responseBody);

            // 解析响应
            Map<String, Object> responseMap = objectMapper.readValue(responseBody, Map.class);
            Integer code = (Integer) responseMap.get("code");
            
            if (code != null && code == 0) {
                log.info("文档解析成功，datasetId: {}, documentIds: {}", datasetId, documentIds);
                return true;
            } else {
                String message = (String) responseMap.get("message");
                log.error("RAGFlow API调用失败，响应码: {}, 消息: {}, 响应内容: {}", code, message, responseBody);
                throw new RenException(ErrorCode.RAG_API_PARSE_FAILED, "RAGFlow API调用失败: " + message);
            }

        } catch (IOException e) {
            log.error("解析RAGFlow API响应失败: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_PARSE_FAILED, "解析RAGFlow响应失败: " + e.getMessage());
        } catch (HttpClientErrorException e) {
            log.error("RAGFlow API调用失败 - HTTP错误: {}, 状态码: {}, 响应内容: {}", 
                e.getMessage(), e.getStatusCode(), e.getResponseBodyAsString(), e);
            throw new RenException(ErrorCode.RAG_API_PARSE_FAILED, "解析RAGFlow文档失败: " + e.getMessage());
        } catch (HttpServerErrorException e) {
            log.error("RAGFlow API调用失败 - 服务器错误: {}, 状态码: {}, 响应内容: {}", 
                e.getMessage(), e.getStatusCode(), e.getResponseBodyAsString(), e);
            throw new RenException(ErrorCode.RAG_API_PARSE_FAILED, "解析RAGFlow文档失败: " + e.getMessage());
        } catch (ResourceAccessException e) {
            log.error("RAGFlow API调用失败 - 网络连接错误: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_PARSE_FAILED, "解析RAGFlow文档失败: 网络连接错误 - " + e.getMessage());
        } catch (Exception e) {
            log.error("解析文档失败: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_PARSE_FAILED, "解析文档失败: " + e.getMessage());
        } finally {
            log.info("=== 解析文档操作结束 ===");
        }
    }

    @Override
    public Map<String, Object> addChunk(String datasetId, String documentId, String content, 
                                       List<String> importantKeywords, List<String> questions) {
        if (StringUtils.isBlank(datasetId) || StringUtils.isBlank(documentId) || StringUtils.isBlank(content)) {
            throw new RenException(ErrorCode.PARAMS_GET_ERROR, "datasetId、documentId和content不能为空");
        }

        log.info("=== 开始添加切片 ===");
        log.info("datasetId: {}, documentId: {}, content长度: {}", datasetId, documentId, content.length());

        try {
            // 获取RAG配置
            Map<String, Object> ragConfig = getRAGConfig();
            String baseUrl = (String) ragConfig.get("base_url");
            String apiKey = (String) ragConfig.get("api_key");

            // 构建请求URL - 根据RAGFlow API文档，添加切片的接口
            String url = baseUrl + "/api/v1/datasets/" + datasetId + "/documents/" + documentId + "/chunks";
            log.debug("请求URL: {}", url);

            // 构建请求体
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("content", content);
            
            if (importantKeywords != null && !importantKeywords.isEmpty()) {
                requestBody.put("important_keywords", importantKeywords);
            }
            
            if (questions != null && !questions.isEmpty()) {
                requestBody.put("questions", questions);
            }
            
            log.debug("请求体: {}", requestBody);

            // 设置请求头
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.set("Authorization", "Bearer " + apiKey);

            HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(requestBody, headers);

            // 发送POST请求
            log.info("发送POST请求到RAGFlow API添加切片...");
            ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.POST, requestEntity, String.class);

            log.info("RAGFlow API响应状态码: {}", response.getStatusCode());

            if (!response.getStatusCode().is2xxSuccessful()) {
                log.error("RAGFlow API调用失败，状态码: {}, 响应内容: {}", response.getStatusCode(), response.getBody());
                throw new RenException(ErrorCode.RAG_API_OPERATION_FAILED, "添加切片失败");
            }

            String responseBody = response.getBody();
            log.debug("RAGFlow API响应内容: {}", responseBody);

            // 解析响应
            Map<String, Object> responseMap = objectMapper.readValue(responseBody, Map.class);
            Integer code = (Integer) responseMap.get("code");
            
            if (code != null && code == 0) {
                log.info("切片添加成功，datasetId: {}, documentId: {}", datasetId, documentId);
                return responseMap;
            } else {
                String message = (String) responseMap.get("message");
                log.error("RAGFlow API调用失败，响应码: {}, 消息: {}, 响应内容: {}", code, message, responseBody);
                throw new RenException(ErrorCode.RAG_API_OPERATION_FAILED, "RAGFlow API调用失败: " + message);
            }

        } catch (IOException e) {
            log.error("解析RAGFlow API响应失败: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_OPERATION_FAILED, "解析RAGFlow响应失败: " + e.getMessage());
        } catch (HttpClientErrorException e) {
            log.error("RAGFlow API调用失败 - HTTP错误: {}, 状态码: {}, 响应内容: {}", 
                e.getMessage(), e.getStatusCode(), e.getResponseBodyAsString(), e);
            throw new RenException(ErrorCode.RAG_API_OPERATION_FAILED, "添加切片失败: " + e.getMessage());
        } catch (HttpServerErrorException e) {
            log.error("RAGFlow API调用失败 - 服务器错误: {}, 状态码: {}, 响应内容: {}", 
                e.getMessage(), e.getStatusCode(), e.getResponseBodyAsString(), e);
            throw new RenException(ErrorCode.RAG_API_OPERATION_FAILED, "添加切片失败: " + e.getMessage());
        } catch (ResourceAccessException e) {
            log.error("RAGFlow API调用失败 - 网络连接错误: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_OPERATION_FAILED, "添加切片失败: 网络连接错误 - " + e.getMessage());
        } catch (Exception e) {
            log.error("添加切片失败: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_OPERATION_FAILED, "添加切片失败: " + e.getMessage());
        } finally {
            log.info("=== 添加切片操作结束 ===");
        }
    }

    @Override
    public Map<String, Object> listChunks(String datasetId, String documentId, String keywords, 
                                        Integer page, Integer pageSize, String chunkId) {
        if (StringUtils.isBlank(datasetId) || StringUtils.isBlank(documentId)) {
            throw new RenException(ErrorCode.PARAMS_GET_ERROR, "datasetId和documentId不能为空");
        }

        log.info("=== 开始列出切片 ===");
        log.info("datasetId: {}, documentId: {}, keywords: {}, page: {}, pageSize: {}, chunkId: {}", 
                datasetId, documentId, keywords, page, pageSize, chunkId);

        try {
            // 获取RAG配置
            Map<String, Object> ragConfig = getRAGConfig();
            String baseUrl = (String) ragConfig.get("base_url");
            String apiKey = (String) ragConfig.get("api_key");

            // 构建请求URL - 根据RAGFlow API文档，列出切片的接口
            StringBuilder urlBuilder = new StringBuilder();
            urlBuilder.append(baseUrl).append("/api/v1/datasets/").append(datasetId)
                     .append("/documents/").append(documentId).append("/chunks");
            
            // 添加查询参数
            List<String> params = new ArrayList<>();
            
            if (StringUtils.isNotBlank(keywords)) {
                params.add("keywords=" + keywords);
            }
            
            if (page != null && page > 0) {
                params.add("page=" + page);
            }
            
            if (pageSize != null && pageSize > 0) {
                params.add("page_size=" + pageSize);
            }
            
            if (StringUtils.isNotBlank(chunkId)) {
                params.add("id=" + chunkId);
            }
            
            if (!params.isEmpty()) {
                urlBuilder.append("?").append(String.join("&", params));
            }
            
            String url = urlBuilder.toString();
            log.debug("请求URL: {}", url);

            // 设置请求头
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.set("Authorization", "Bearer " + apiKey);

            HttpEntity<String> requestEntity = new HttpEntity<>(headers);

            // 发送GET请求
            log.info("发送GET请求到RAGFlow API列出切片...");
            ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.GET, requestEntity, String.class);

            log.info("RAGFlow API响应状态码: {}", response.getStatusCode());

            if (!response.getStatusCode().is2xxSuccessful()) {
                log.error("RAGFlow API调用失败，状态码: {}, 响应内容: {}", response.getStatusCode(), response.getBody());
                throw new RenException(ErrorCode.RAG_API_OPERATION_FAILED, "列出切片失败");
            }

            String responseBody = response.getBody();
            log.debug("RAGFlow API响应内容: {}", responseBody);

            // 解析响应
            Map<String, Object> responseMap = objectMapper.readValue(responseBody, Map.class);
            Integer code = (Integer) responseMap.get("code");
            
            if (code != null && code == 0) {
                log.info("切片列表获取成功，datasetId: {}, documentId: {}", datasetId, documentId);
                
                // 解析切片数据并格式化返回
                return parseChunkListResponse(responseMap);
            } else {
                String message = (String) responseMap.get("message");
                log.error("RAGFlow API调用失败，响应码: {}, 消息: {}, 响应内容: {}", code, message, responseBody);
                throw new RenException(ErrorCode.RAG_API_OPERATION_FAILED, "RAGFlow API调用失败: " + message);
            }

        } catch (IOException e) {
            log.error("解析RAGFlow API响应失败: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_OPERATION_FAILED, "解析RAGFlow响应失败: " + e.getMessage());
        } catch (HttpClientErrorException e) {
            log.error("RAGFlow API调用失败 - HTTP错误: {}, 状态码: {}, 响应内容: {}", 
                e.getMessage(), e.getStatusCode(), e.getResponseBodyAsString(), e);
            throw new RenException(ErrorCode.RAG_API_OPERATION_FAILED, "列出切片失败: " + e.getMessage());
        } catch (HttpServerErrorException e) {
            log.error("RAGFlow API调用失败 - 服务器错误: {}, 状态码: {}, 响应内容: {}", 
                e.getMessage(), e.getStatusCode(), e.getResponseBodyAsString(), e);
            throw new RenException(ErrorCode.RAG_API_OPERATION_FAILED, "列出切片失败: " + e.getMessage());
        } catch (ResourceAccessException e) {
            log.error("RAGFlow API调用失败 - 网络连接错误: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_OPERATION_FAILED, "列出切片失败: 网络连接错误 - " + e.getMessage());
        } catch (Exception e) {
            log.error("列出切片失败: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_OPERATION_FAILED, "列出切片失败: " + e.getMessage());
        } finally {
            log.info("=== 列出切片操作结束 ===");
        }
    }

    /**
     * 解析RAGFlow API返回的切片列表响应
     */
    private Map<String, Object> parseChunkListResponse(Map<String, Object> responseMap) {
        Map<String, Object> result = new HashMap<>();
        List<Map<String, Object>> chunkList = new ArrayList<>();
        long totalCount = 0;

        try {
            // 首先检查是否有data字段
            Object dataObj = responseMap.get("data");
            if (dataObj instanceof Map) {
                Map<String, Object> dataMap = (Map<String, Object>) dataObj;
                
                // 解析切片列表 - 支持多种可能的字段名
                Object chunksObj = null;
                
                // 支持多种可能的切片列表字段名
                String[] possibleChunkFields = {"chunks", "items", "list", "data", "docs"};
                
                for (String fieldName : possibleChunkFields) {
                    if (dataMap.containsKey(fieldName) && dataMap.get(fieldName) instanceof List) {
                        chunksObj = dataMap.get(fieldName);
                        log.debug("使用字段名'{}'获取切片列表", fieldName);
                        break;
                    }
                }
                
                // 如果标准字段不存在，尝试自动检测
                if (chunksObj == null) {
                    for (Map.Entry<String, Object> entry : dataMap.entrySet()) {
                        if (entry.getValue() instanceof List) {
                            List<?> list = (List<?>) entry.getValue();
                            if (!list.isEmpty() && list.get(0) instanceof Map) {
                                chunksObj = entry.getValue();
                                log.warn("自动检测到切片列表字段: '{}'，建议检查RAGFlow API文档", entry.getKey());
                                break;
                            }
                        }
                    }
                }
                
                if (chunksObj instanceof List) {
                    List<Map<String, Object>> rawChunkList = (List<Map<String, Object>>) chunksObj;
                    
                    for (Map<String, Object> chunkMap : rawChunkList) {
                        Map<String, Object> formattedChunk = formatChunkData(chunkMap);
                        if (formattedChunk != null) {
                            chunkList.add(formattedChunk);
                        }
                    }
                }
                
                // 解析总数 - 支持多种可能的字段名
                Object totalObj = null;
                String[] possibleTotalFields = {"total", "totalCount", "total_count", "count"};
                
                for (String fieldName : possibleTotalFields) {
                    if (dataMap.containsKey(fieldName)) {
                        totalObj = dataMap.get(fieldName);
                        log.debug("使用字段名'{}'获取总数", fieldName);
                        break;
                    }
                }
                
                if (totalObj instanceof Integer) {
                    totalCount = ((Integer) totalObj).longValue();
                } else if (totalObj instanceof Long) {
                    totalCount = (Long) totalObj;
                } else if (totalObj instanceof String) {
                    try {
                        totalCount = Long.parseLong((String) totalObj);
                    } catch (NumberFormatException e) {
                        log.warn("无法解析总数字段: {}", totalObj);
                    }
                }
                
                // 如果没有找到总数，使用切片列表的大小
                if (totalCount == 0 && !chunkList.isEmpty()) {
                    totalCount = chunkList.size();
                }
            } else {
                log.warn("RAGFlow API响应缺少data字段，尝试直接解析响应");
                
                // 如果没有data字段，尝试直接解析响应
                Object chunksObj = null;
                String[] possibleChunkFields = {"chunks", "items", "list", "data", "docs"};
                
                for (String fieldName : possibleChunkFields) {
                    if (responseMap.containsKey(fieldName) && responseMap.get(fieldName) instanceof List) {
                        chunksObj = responseMap.get(fieldName);
                        log.debug("使用字段名'{}'获取切片列表", fieldName);
                        break;
                    }
                }
                
                if (chunksObj instanceof List) {
                    List<Map<String, Object>> rawChunkList = (List<Map<String, Object>>) chunksObj;
                    
                    for (Map<String, Object> chunkMap : rawChunkList) {
                        Map<String, Object> formattedChunk = formatChunkData(chunkMap);
                        if (formattedChunk != null) {
                            chunkList.add(formattedChunk);
                        }
                    }
                }
                
                // 解析总数
                Object totalObj = responseMap.get("total");
                if (totalObj instanceof Integer) {
                    totalCount = ((Integer) totalObj).longValue();
                } else if (totalObj instanceof Long) {
                    totalCount = (Long) totalObj;
                }
                
                if (totalCount == 0 && !chunkList.isEmpty()) {
                    totalCount = chunkList.size();
                }
            }
            
        } catch (Exception e) {
            log.error("解析切片列表响应失败: {}", e.getMessage(), e);
        }
        
        result.put("list", chunkList);
        result.put("total", totalCount);
        
        log.debug("解析后的切片列表: {} 条记录", chunkList.size());
        return result;
    }

    /**
     * 格式化切片数据
     */
    private Map<String, Object> formatChunkData(Map<String, Object> chunkMap) {
        if (chunkMap == null || chunkMap.isEmpty()) {
            return null;
        }
        
        Map<String, Object> formattedChunk = new HashMap<>();
        
        try {
            // 提取切片ID - 支持多种可能的字段名
            String chunkId = extractChunkId(chunkMap);
            if (StringUtils.isBlank(chunkId)) {
                log.warn("切片数据缺少ID字段，跳过处理: {}", chunkMap);
                return null;
            }
            formattedChunk.put("id", chunkId);
            
            // 提取切片内容 - 支持多种可能的字段名
            String content = extractChunkContent(chunkMap);
            formattedChunk.put("content", content != null ? content : "");
            
            // 提取重要关键词 - 支持多种可能的字段名
            List<String> importantKeywords = extractImportantKeywords(chunkMap);
            formattedChunk.put("important_keywords", importantKeywords);
            
            // 提取问题列表 - 支持多种可能的字段名
            List<String> questions = extractQuestions(chunkMap);
            formattedChunk.put("questions", questions);
            
            // 提取创建时间 - 支持多种可能的字段名
            String createTime = extractCreateTime(chunkMap);
            formattedChunk.put("create_time", createTime != null ? createTime : "");
            
        } catch (Exception e) {
            log.error("格式化切片数据失败: {}", e.getMessage(), e);
            return null;
        }
        
        return formattedChunk;
    }

    /**
     * 提取切片ID
     */
    private String extractChunkId(Map<String, Object> chunkMap) {
        String[] possibleIdFields = {"id", "chunk_id", "chunkId", "chunkId"};
        
        for (String fieldName : possibleIdFields) {
            Object value = chunkMap.get(fieldName);
            if (value != null && value instanceof String && StringUtils.isNotBlank((String) value)) {
                return (String) value;
            }
        }
        
        return null;
    }

    /**
     * 提取切片内容
     */
    private String extractChunkContent(Map<String, Object> chunkMap) {
        String[] possibleContentFields = {"content", "text", "chunk_content", "chunkContent"};
        
        for (String fieldName : possibleContentFields) {
            Object value = chunkMap.get(fieldName);
            if (value != null && value instanceof String && StringUtils.isNotBlank((String) value)) {
                return (String) value;
            }
        }
        
        return null;
    }

    /**
     * 提取重要关键词
     */
    private List<String> extractImportantKeywords(Map<String, Object> chunkMap) {
        String[] possibleKeywordFields = {"important_keywords", "keywords", "importantKeywords", "key_words"};
        
        for (String fieldName : possibleKeywordFields) {
            Object value = chunkMap.get(fieldName);
            if (value instanceof List) {
                List<?> list = (List<?>) value;
                List<String> keywords = new ArrayList<>();
                for (Object item : list) {
                    if (item instanceof String && StringUtils.isNotBlank((String) item)) {
                        keywords.add((String) item);
                    }
                }
                return keywords;
            } else if (value instanceof String && StringUtils.isNotBlank((String) value)) {
                // 如果是逗号分隔的字符串，分割成列表
                String[] parts = ((String) value).split(",");
                List<String> keywords = new ArrayList<>();
                for (String part : parts) {
                    String trimmed = part.trim();
                    if (StringUtils.isNotBlank(trimmed)) {
                        keywords.add(trimmed);
                    }
                }
                return keywords;
            }
        }
        
        return new ArrayList<>();
    }

    /**
     * 提取问题列表
     */
    private List<String> extractQuestions(Map<String, Object> chunkMap) {
        String[] possibleQuestionFields = {"questions", "question_list", "questionList", "qas"};
        
        for (String fieldName : possibleQuestionFields) {
            Object value = chunkMap.get(fieldName);
            if (value instanceof List) {
                List<?> list = (List<?>) value;
                List<String> questions = new ArrayList<>();
                for (Object item : list) {
                    if (item instanceof String && StringUtils.isNotBlank((String) item)) {
                        questions.add((String) item);
                    }
                }
                return questions;
            } else if (value instanceof String && StringUtils.isNotBlank((String) value)) {
                // 如果是逗号分隔的字符串，分割成列表
                String[] parts = ((String) value).split(",");
                List<String> questions = new ArrayList<>();
                for (String part : parts) {
                    String trimmed = part.trim();
                    if (StringUtils.isNotBlank(trimmed)) {
                        questions.add(trimmed);
                    }
                }
                return questions;
            }
        }
        
        return new ArrayList<>();
    }

    /**
     * 提取创建时间
     */
    private String extractCreateTime(Map<String, Object> chunkMap) {
        String[] possibleTimeFields = {"create_time", "created_at", "createTime", "timestamp"};
        
        for (String fieldName : possibleTimeFields) {
            Object value = chunkMap.get(fieldName);
            if (value != null) {
                if (value instanceof String && StringUtils.isNotBlank((String) value)) {
                    return (String) value;
                } else if (value instanceof Long) {
                    // 如果是时间戳，转换为字符串
                    return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date((Long) value));
                }
            }
        }
        
        return null;
    }

    @Override
    public Map<String, Object> retrievalTest(String question, List<String> datasetIds, List<String> documentIds, 
                                           Integer page, Integer pageSize, Float similarityThreshold, 
                                           Float vectorSimilarityWeight, Integer topK, String rerankId, 
                                           Boolean keyword, Boolean highlight, List<String> crossLanguages, 
                                           Map<String, Object> metadataCondition) {
        
        log.info("=== 开始召回测试 ===");
        log.info("问题: {}, 数据集ID: {}, 文档ID: {}, 页码: {}, 每页数量: {}", 
                question, datasetIds, documentIds, page, pageSize);

        try {
            // 获取RAG配置
            Map<String, Object> ragConfig = getRAGConfig();
            String baseUrl = (String) ragConfig.get("base_url");
            String apiKey = (String) ragConfig.get("api_key");

            // 构建请求URL
            String url = baseUrl + "/api/v1/retrieval";
            log.debug("请求URL: {}", url);

            // 构建请求体
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("question", question);
            
            if (datasetIds != null && !datasetIds.isEmpty()) {
                requestBody.put("dataset_ids", datasetIds);
            }
            
            if (documentIds != null && !documentIds.isEmpty()) {
                requestBody.put("document_ids", documentIds);
            }
            
            if (page != null && page > 0) {
                requestBody.put("page", page);
            }
            
            if (pageSize != null && pageSize > 0) {
                requestBody.put("page_size", pageSize);
            }
            
            if (similarityThreshold != null) {
                requestBody.put("similarity_threshold", similarityThreshold);
            }
            
            if (vectorSimilarityWeight != null) {
                requestBody.put("vector_similarity_weight", vectorSimilarityWeight);
            }
            
            if (topK != null && topK > 0) {
                requestBody.put("top_k", topK);
            }
            
            if (rerankId != null) {
                requestBody.put("rerank_id", rerankId);
            }
            
            if (keyword != null) {
                requestBody.put("keyword", keyword);
            }
            
            if (highlight != null) {
                requestBody.put("highlight", highlight);
            }
            
            if (crossLanguages != null && !crossLanguages.isEmpty()) {
                requestBody.put("cross_languages", crossLanguages);
            }
            
            if (metadataCondition != null) {
                requestBody.put("metadata_condition", metadataCondition);
            }

            log.debug("请求体: {}", requestBody);

            // 设置请求头
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.set("Authorization", "Bearer " + apiKey);

            HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(requestBody, headers);

            // 发送POST请求
            log.info("发送POST请求到RAGFlow API进行召回测试...");
            ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.POST, requestEntity, String.class);

            log.info("RAGFlow API响应状态码: {}", response.getStatusCode());

            if (!response.getStatusCode().is2xxSuccessful()) {
                log.error("RAGFlow API调用失败，状态码: {}, 响应内容: {}", response.getStatusCode(), response.getBody());
                throw new RenException(ErrorCode.RAG_API_QUERY_FAILED);
            }

            String responseBody = response.getBody();
            log.debug("RAGFlow API响应内容: {}", responseBody);

            // 解析响应
            Map<String, Object> responseMap = objectMapper.readValue(responseBody, Map.class);
            Integer code = (Integer) responseMap.get("code");
            
            if (code != null && code == 0) {
                Object dataObj = responseMap.get("data");
                if (dataObj instanceof Map) {
                    Map<String, Object> result = (Map<String, Object>) dataObj;
                    log.info("召回测试成功，返回 {} 条切片", result.get("total"));
                    return result;
                } else {
                    log.error("RAGFlow API响应格式错误，data字段不是Map类型: {}", dataObj);
                    throw new RenException(ErrorCode.RAG_API_QUERY_FAILED, "RAGFlow API响应格式错误");
                }
            } else {
                String message = (String) responseMap.get("message");
                log.error("RAGFlow API调用失败，响应码: {}, 错误信息: {}", code, message);
                throw new RenException(ErrorCode.RAG_API_QUERY_FAILED, "RAGFlow API调用失败: " + message);
            }

        } catch (IOException e) {
            log.error("解析RAGFlow API响应失败: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_QUERY_FAILED, "解析RAGFlow响应失败: " + e.getMessage());
        } catch (HttpClientErrorException e) {
            log.error("RAGFlow API调用失败 - HTTP错误: {}, 状态码: {}, 响应内容: {}", 
                e.getMessage(), e.getStatusCode(), e.getResponseBodyAsString(), e);
            throw new RenException(ErrorCode.RAG_API_QUERY_FAILED, "召回测试失败: " + e.getMessage());
        } catch (HttpServerErrorException e) {
            log.error("RAGFlow API调用失败 - 服务器错误: {}, 状态码: {}, 响应内容: {}", 
                e.getMessage(), e.getStatusCode(), e.getResponseBodyAsString(), e);
            throw new RenException(ErrorCode.RAG_API_QUERY_FAILED, "召回测试失败: " + e.getMessage());
        } catch (ResourceAccessException e) {
            log.error("RAGFlow API调用失败 - 网络连接错误: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_QUERY_FAILED, "召回测试失败: 网络连接错误 - " + e.getMessage());
        } catch (Exception e) {
            log.error("召回测试失败: {}", e.getMessage(), e);
            throw new RenException(ErrorCode.RAG_API_QUERY_FAILED, "召回测试失败: " + e.getMessage());
        } finally {
            log.info("=== 召回测试操作结束 ===");
        }
    }
}