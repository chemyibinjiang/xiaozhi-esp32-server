package xiaozhi.modules.agent.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import xiaozhi.common.constant.Constant;
import xiaozhi.common.exception.RenException;
import xiaozhi.common.utils.ConvertUtils;
import xiaozhi.modules.agent.dao.AgentVoicePrintDao;
import xiaozhi.modules.agent.dto.AgentVoicePrintSaveDTO;
import xiaozhi.modules.agent.dto.AgentVoicePrintUpdateDTO;
import xiaozhi.modules.agent.entity.AgentVoicePrintEntity;
import xiaozhi.modules.agent.service.AgentChatAudioService;
import xiaozhi.modules.agent.service.AgentVoicePrintService;
import xiaozhi.modules.agent.vo.AgentVoicePrintVO;
import xiaozhi.modules.sys.service.SysParamsService;

import java.beans.Transient;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.List;

/**
 * @author zjy
 */
@Service
@AllArgsConstructor
@Slf4j
public class AgentVoicePrintServiceImpl extends ServiceImpl<AgentVoicePrintDao, AgentVoicePrintEntity>
        implements AgentVoicePrintService {
    private final AgentChatAudioService agentChatAudioService;
    private final RestTemplate restTemplate;
    private final SysParamsService sysParamsService;


    @Override
    @Transient
    public boolean insert(AgentVoicePrintSaveDTO dto) {
        // 获取音频Id
        String audioId = dto.getAudioId();
        ByteArrayResource resource = getVoicePrintAudioWAV(audioId);

        // 保存声纹信息
        AgentVoicePrintEntity entity = ConvertUtils.sourceToTarget(dto, AgentVoicePrintEntity.class);
        int insert = baseMapper.insert(entity);
        if(insert != 1){
            return false;
        }
        registerVoicePrint(entity.getId(), resource);
        return true;
    }

    @Override
    public boolean delete(String voicePrintId) {
        int insert = baseMapper.deleteById(voicePrintId);
        if(insert != 1){
            throw new RenException("声纹删除失败");
        }
        URI uri = getVoicePrintURI();
        String baseUrl = getBaseUrl(uri);
        String requestUrl =  baseUrl + "/voiceprint/" + voicePrintId;
        // 创建请求头
        HttpHeaders headers = new HttpHeaders();
        headers.set("Authorization", getAuthorization(uri));
        // 创建请求体
        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(headers);

        // 发送 POST 请求
        ResponseEntity<String> response = restTemplate.exchange(requestUrl, HttpMethod.DELETE, requestEntity, String.class);
        if (response.getStatusCode() != HttpStatus.OK) {
            throw new RenException("声纹保存失败");
        }
        // 检查响应内容
        String responseBody = response.getBody();
        if(responseBody == null || !responseBody.contains("true")){
            throw new RenException("声纹保存失败");
        }
        return true;
    }

    @Override
    public List<AgentVoicePrintVO> list(String agentId) {
        List<AgentVoicePrintEntity> list = baseMapper.selectList(new LambdaQueryWrapper<AgentVoicePrintEntity>()
                .eq(AgentVoicePrintEntity::getAgentId, agentId));
        return list.stream().map(entity -> {
            // 遍历转换成AgentVoicePrintVO类型
           return ConvertUtils.sourceToTarget(entity, AgentVoicePrintVO.class);
        }).toList();

    }

    @Override
    public boolean update(AgentVoicePrintUpdateDTO dto) {
        // 获取音频Id
        String audioId = dto.getAudioId();
        // 如果有新的音频，则注册新的声纹
        if (!StringUtils.isEmpty(audioId)) {
            ByteArrayResource resource = getVoicePrintAudioWAV(audioId);
            registerVoicePrint(dto.getId(),resource);
        }
        AgentVoicePrintEntity entity = ConvertUtils.sourceToTarget(dto, AgentVoicePrintEntity.class);
        baseMapper.updateById(entity);
        return true;
    }

    /**
     * 获取生纹接口URI对象
     *
     * @return URI对象
     */
    private URI getVoicePrintURI() {
        // 获取声纹接口地址
        String voicePrint = sysParamsService.getValue(Constant.SERVER_VOICE_PRINT, true);
        try {
            return new URI(voicePrint);
        } catch (URISyntaxException e) {
            log.error("路径格式不正确路径：{}，\n错误信息:{}", voicePrint, e.getMessage());
            throw new RuntimeException("声纹接口的地址存在错误，请进入参数管理修改声纹接口地址");
        }
    }

    /**
     * 获取声纹地址基础路径
     * @param uri 声纹地址uri
     * @return 基础路径
     */
    private String getBaseUrl(URI uri) {
        String protocol = uri.getScheme();
        String host = uri.getHost();
        int port = uri.getPort();
        return "%s://%s:%s".formatted(protocol,host,port);
    }

    /**
     * 获取验证Authorization
     *
     * @param uri 声纹地址uri
     * @return Authorization值
     */
    private String getAuthorization(URI uri) {
        // 获取参数
        String query = uri.getQuery();
        // 获取aes加密密钥
        String str = "key=";
        return "Bearer " + query.substring(query.indexOf(str) + str.length());
    }

    /**
     * 获取声纹音频资源数据
     * @param audioId  音频Id
     * @return 声纹音频资源数据
     */
    private ByteArrayResource getVoicePrintAudioWAV(String audioId) {
        // 获取到音频数据
        byte[] audio = agentChatAudioService.getAudio(audioId);
        // 如果音频数据为空的直接报错不进行下去
        if (audio == null || audio.length == 0) {
            throw new RenException("音频数据是空的请检查上传数据");
        }
        // 将字节数组包装为资源，返回
        return new ByteArrayResource(audio) {
            @Override
            public String getFilename() {
                return "VoicePrint.WAV"; // 设置文件名
            }
        };
    }

    /**
     *  发送注册声纹http请求
     * @param id 声纹id
     * @param resource 声纹音频资源
     */
    private void registerVoicePrint(String id, ByteArrayResource resource) {
        // 处理声纹接口地址，获取前缀
        URI uri = getVoicePrintURI();
        String baseUrl = getBaseUrl(uri);
        String requestUrl =  baseUrl + "/voiceprint/register";
        // 创建请求体
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("speaker_id", id);
        body.add("file", resource);

        // 创建请求头
        HttpHeaders headers = new HttpHeaders();
        headers.set("Authorization", getAuthorization(uri));
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        // 创建请求体
        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);
        // 发送 POST 请求
        ResponseEntity<String> response = restTemplate.postForEntity(requestUrl, requestEntity, String.class);

        if (response.getStatusCode() != HttpStatus.OK) {
            throw new RenException("声纹保存失败");
        }
        // 检查响应内容
        String responseBody = response.getBody();
        if(responseBody == null || !responseBody.contains("true")){
            throw new RenException("声纹保存失败");
        }
    }
}
