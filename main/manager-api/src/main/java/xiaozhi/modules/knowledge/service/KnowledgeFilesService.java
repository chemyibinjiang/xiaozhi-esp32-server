package xiaozhi.modules.knowledge.service;

import java.util.List;
import java.util.Map;

import org.springframework.web.multipart.MultipartFile;

import xiaozhi.common.page.PageData;
import xiaozhi.modules.knowledge.dto.KnowledgeFilesDTO;

/**
 * 知识库文档服务接口
 */
public interface KnowledgeFilesService {

    /**
     * 分页查询文档列表
     * 
     * @param knowledgeFilesDTO 查询条件
     * @param page 页码
     * @param limit 每页数量
     * @return 分页数据
     */
    PageData<KnowledgeFilesDTO> getPageList(KnowledgeFilesDTO knowledgeFilesDTO, Integer page, Integer limit);

    /**
     * 根据ID获取文档详情
     * 
     * @param id 文档ID
     * @return 文档详情
     */
    KnowledgeFilesDTO getById(String id);

    /**
     * 根据文档ID获取文档详情
     * 
     * @param documentId 文档ID
     * @return 文档详情
     */
    KnowledgeFilesDTO getByDocumentId(String documentId);

    /**
     * 根据文档ID和知识库ID获取文档详情
     * 
     * @param documentId 文档ID
     * @param datasetId 知识库ID
     * @return 文档详情
     */
    KnowledgeFilesDTO getByDocumentId(String documentId, String datasetId);

    /**
     * 上传文档到知识库
     * 
     * @param datasetId 知识库ID
     * @param file 上传的文件
     * @param name 文档名称
     * @param metaFields 元数据字段
     * @param chunkMethod 分块方法
     * @param parserConfig 解析器配置
     * @return 上传的文档信息
     */
    KnowledgeFilesDTO uploadDocument(String datasetId, MultipartFile file, String name, 
                                     Map<String, Object> metaFields, String chunkMethod, 
                                     Map<String, Object> parserConfig);

    /**
     * 更新文档配置
     * 
     * @param knowledgeFilesDTO 文档信息
     * @return 更新的文档信息
     */
    KnowledgeFilesDTO update(KnowledgeFilesDTO knowledgeFilesDTO);

    /**
     * 根据ID删除文档
     * 
     * @param id 文档ID
     */
    void delete(String id);

    /**
     * 根据文档ID删除文档
     * 
     * @param documentId 文档ID
     */
    void deleteByDocumentId(String documentId);

    /**
     * 根据文档ID和知识库ID删除文档
     * 
     * @param documentId 文档ID
     * @param datasetId 知识库ID
     */
    void deleteByDocumentId(String documentId, String datasetId);

    /**
     * 批量删除文档
     * 
     * @param ids 文档ID列表
     */
    void deleteBatch(List<String> ids);


    /**
     * 获取RAG配置信息
     * 
     * @param ragModelId RAG模型配置ID
     * @return RAG配置信息
     */
    Map<String, Object> getRAGConfig(String ragModelId);
    
    /**
     * 获取默认RAG配置信息
     * 
     * @return 默认RAG配置信息
     */
    Map<String, Object> getDefaultRAGConfig();

    /**
     * 解析文档（切块）
     * 
     * @param datasetId 知识库ID
     * @param documentIds 文档ID列表
     * @return 解析结果
     */
    boolean parseDocuments(String datasetId, List<String> documentIds);

    /**
     * 添加切片到指定文档
     * 
     * @param datasetId 知识库ID
     * @param documentId 文档ID
     * @param content 切片内容
     * @param importantKeywords 重要关键词列表
     * @param questions 问题列表
     * @return 添加的切片信息
     */
    Map<String, Object> addChunk(String datasetId, String documentId, String content, 
                                List<String> importantKeywords, List<String> questions);

    /**
     * 列出指定文档的切片
     * 
     * @param datasetId 知识库ID
     * @param documentId 文档ID
     * @param keywords 关键词过滤
     * @param page 页码
     * @param pageSize 每页数量
     * @param chunkId 切片ID
     * @return 切片列表信息
     */
    Map<String, Object> listChunks(String datasetId, String documentId, String keywords, 
                                  Integer page, Integer pageSize, String chunkId);

    /**
     * 召回测试 - 从指定数据集或文档中检索相关切片
     * 
     * @param question 用户查询或查询关键词
     * @param datasetIds 数据集ID列表
     * @param documentIds 文档ID列表
     * @param page 页码
     * @param pageSize 每页数量
     * @param similarityThreshold 最小相似度阈值
     * @param vectorSimilarityWeight 向量相似度权重
     * @param topK 参与向量余弦计算的切片数量
     * @param rerankId 重排模型ID
     * @param keyword 是否启用关键词匹配
     * @param highlight 是否启用高亮显示
     * @param crossLanguages 跨语言翻译列表
     * @param metadataCondition 元数据过滤条件
     * @return 召回测试结果
     */
    Map<String, Object> retrievalTest(String question, List<String> datasetIds, List<String> documentIds, 
                                     Integer page, Integer pageSize, Float similarityThreshold, 
                                     Float vectorSimilarityWeight, Integer topK, String rerankId, 
                                     Boolean keyword, Boolean highlight, List<String> crossLanguages, 
                                     Map<String, Object> metadataCondition);
}