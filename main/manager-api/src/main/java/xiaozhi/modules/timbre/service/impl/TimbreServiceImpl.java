package xiaozhi.modules.timbre.service.impl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import xiaozhi.common.exception.ErrorCode;
import xiaozhi.common.utils.MessageUtils;

import org.apache.commons.lang3.StringUtils;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;

import cn.hutool.core.collection.CollectionUtil;
import lombok.AllArgsConstructor;
import xiaozhi.common.constant.Constant;
import xiaozhi.common.page.PageData;
import xiaozhi.common.redis.RedisKeys;
import xiaozhi.common.redis.RedisUtils;
import xiaozhi.common.service.impl.BaseServiceImpl;
import xiaozhi.common.utils.ConvertUtils;
import xiaozhi.modules.model.dto.VoiceDTO;
import xiaozhi.modules.timbre.dao.TimbreDao;
import xiaozhi.modules.timbre.dto.TimbreDataDTO;
import xiaozhi.modules.timbre.dto.TimbrePageDTO;
import xiaozhi.modules.timbre.entity.TimbreEntity;
import xiaozhi.modules.timbre.service.TimbreService;
import xiaozhi.modules.timbre.vo.TimbreDetailsVO;

/**
 * 音色的业务层的实现
 * 
 * @author zjy
 * @since 2025-3-21
 */
@AllArgsConstructor
@Service
public class TimbreServiceImpl extends BaseServiceImpl<TimbreDao, TimbreEntity> implements TimbreService {

    private final TimbreDao timbreDao;
    private final RedisUtils redisUtils;

    @Override
    public PageData<TimbreDetailsVO> page(TimbrePageDTO dto) {
        Map<String, Object> params = new HashMap<String, Object>();
        params.put(Constant.PAGE, dto.getPage());
        params.put(Constant.LIMIT, dto.getLimit());
        IPage<TimbreEntity> page = baseDao.selectPage(
                getPage(params, null, true),
                // 定义查询条件
                new QueryWrapper<TimbreEntity>()
                        // 必须按照ttsID查找
                        .eq("tts_model_id", dto.getTtsModelId())
                        // 如果有音色名字，按照音色名模糊查找
                        .like(StringUtils.isNotBlank(dto.getName()), "name", dto.getName()));

        return getPageData(page, TimbreDetailsVO.class);
    }

    @Override
    public TimbreDetailsVO get(String timbreId) {
        if (StringUtils.isBlank(timbreId)) {
            return null;
        }

        // 先从Redis获取缓存
        String key = RedisKeys.getTimbreDetailsKey(timbreId);
        TimbreDetailsVO cachedDetails = (TimbreDetailsVO) redisUtils.get(key);
        if (cachedDetails != null) {
            return cachedDetails;
        }

        // 如果缓存中没有，则从数据库获取
        TimbreEntity entity = baseDao.selectById(timbreId);
        if (entity == null) {
            return null;
        }

        // 转换为VO对象
        TimbreDetailsVO details = ConvertUtils.sourceToTarget(entity, TimbreDetailsVO.class);

        // 存入Redis缓存
        if (details != null) {
            redisUtils.set(key, details);
        }

        return details;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void save(TimbreDataDTO dto) {
        isTtsModelId(dto.getTtsModelId());
        TimbreEntity timbreEntity = ConvertUtils.sourceToTarget(dto, TimbreEntity.class);
        baseDao.insert(timbreEntity);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void update(String timbreId, TimbreDataDTO dto) {
        isTtsModelId(dto.getTtsModelId());
        TimbreEntity timbreEntity = ConvertUtils.sourceToTarget(dto, TimbreEntity.class);
        timbreEntity.setId(timbreId);
        baseDao.updateById(timbreEntity);
        // 删除缓存
        redisUtils.delete(RedisKeys.getTimbreDetailsKey(timbreId));
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void delete(String[] ids) {
        baseDao.deleteBatchIds(Arrays.asList(ids));
    }

    @Override
    public List<VoiceDTO> getVoiceNames(String ttsModelId, String voiceName) {
        QueryWrapper<TimbreEntity> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("tts_model_id", StringUtils.isBlank(ttsModelId) ? "" : ttsModelId);
        if (StringUtils.isNotBlank(voiceName)) {
            queryWrapper.like("name", voiceName);
        }
        
        List<TimbreEntity> timbreEntities = timbreDao.selectList(queryWrapper);
        if (CollectionUtil.isEmpty(timbreEntities)) {
            return null;
        }

        // 自定义排序：音色编码以S_开头的排在最前面，其他按sort字段升序排序
        timbreEntities.sort((t1, t2) -> {
            // 先判断是否以S_开头
            boolean isT1S_Start = StringUtils.isNotBlank(t1.getTtsVoice()) && t1.getTtsVoice().startsWith("S_");
            boolean isT2S_Start = StringUtils.isNotBlank(t2.getTtsVoice()) && t2.getTtsVoice().startsWith("S_");
            
            if (isT1S_Start && !isT2S_Start) return -1;  // t1以S_开头，排在前面
            if (!isT1S_Start && isT2S_Start) return 1;   // t2以S_开头，排在前面
            
            // 都以S_开头或都不以S_开头，则按sort字段升序排序
            return Long.compare(t1.getSort(), t2.getSort());
        });

        // 排序完成后转换为DTO列表，并为复刻音色添加前缀
        List<VoiceDTO> voiceDTOs = new ArrayList<>();
        for (TimbreEntity entity : timbreEntities) {
            VoiceDTO dto = new VoiceDTO();
            dto.setId(entity.getId());
            // 对于音色编码以S_开头的复刻音色，在名称前加上"克隆："前缀
            String name = entity.getName();
            if (StringUtils.isNotBlank(entity.getTtsVoice()) && entity.getTtsVoice().startsWith("S_") && !name.startsWith("克隆：")) {
                name = MessageUtils.getMessage(ErrorCode.VOICE_CLONE_PREFIX) + name;
            }
            dto.setName(name);
            voiceDTOs.add(dto);
        }
        return voiceDTOs;
    }

    /**
     * 处理是不是tts模型的id
     */
    private void isTtsModelId(String ttsModelId) {
        // 等模型配置那边写好调用方法判断
    }

    @Override
    public String getTimbreNameById(String id) {
        if (StringUtils.isBlank(id)) {
            return null;
        }

        String cachedName = (String) redisUtils.get(RedisKeys.getTimbreNameById(id));

        if (StringUtils.isNotBlank(cachedName)) {
            return cachedName;
        }

        TimbreEntity entity = timbreDao.selectById(id);
        if (entity != null) {
            String name = entity.getName();
            if (StringUtils.isNotBlank(name)) {
                redisUtils.set(RedisKeys.getTimbreNameById(id), name);
            }
            return name;
        }

        return null;
    }

    @Override
    public boolean existsByTtsVoice(String ttsVoice) {
        if (StringUtils.isBlank(ttsVoice)) {
            return false;
        }
        
        QueryWrapper<TimbreEntity> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("tts_voice", ttsVoice);
        
        // 检查数据库中是否存在相同的ttsVoice值
        return timbreDao.exists(queryWrapper);
    }
}