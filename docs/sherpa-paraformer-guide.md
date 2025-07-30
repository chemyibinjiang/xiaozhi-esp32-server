# Sherpa-ONNX Paraformer 模型使用指南

本文档介绍如何在小智 ESP32 服务器中使用 Paraformer 模型进行中文语音识别。Paraformer 有多个版本，本指南重点介绍适合低性能设备（如 RK3566）的小尺寸版本。

## 为什么选择 Paraformer？

Paraformer 是阿里达摩院开发的语音识别模型系列，支持多种语言和尺寸。对于性能较低的设备（如 RK3566），我们推荐使用中文小尺寸版本：

- **多种尺寸可选**：从 78MB 的小模型到 600MB+ 的大模型
- **速度快**：优化的模型结构，推理速度快
- **资源占用灵活**：可根据设备性能选择合适的版本

## Paraformer 模型特点

- **多语言支持**：除中文外，还有英文、日文等版本
- **支持 INT8 量化**：通过量化技术减小模型体积
- **CPU 友好**：即使在纯 CPU 环境下也能高效运行
- **低延迟**：适合实时对话场景

**注意**：本指南主要介绍中文版本（paraformer-zh 系列），这是小智 ESP32 项目的主要使用场景。

## 如何找到合适的模型

### 模型来源渠道

1. **GitHub Releases（推荐）**
   - [Sherpa-ONNX 官方发布页](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)
   - 优点：模型经过验证，格式标准，下载稳定
   - 缺点：需要科学上网

2. **ModelScope（阿里巴巴）**
   - [ModelScope 搜索](https://modelscope.cn/models?name=sherpa-onnx&page=1)
   - 优点：国内访问快，无需科学上网
   - 缺点：部分模型格式不标准，需要仔细筛选

3. **HuggingFace**
   - [HuggingFace 搜索](https://huggingface.co/models?search=sherpa-onnx)
   - 优点：模型丰富，社区活跃
   - 缺点：需要科学上网，下载速度慢

### 判断模型是否可用

**必须满足以下条件**：

1. **文件格式要求**
   ```
   模型目录/
   ├── model.int8.onnx    # INT8 量化的 ONNX 模型（必需）
   ├── tokens.txt         # 词汇表文件（必需）
   └── README.md          # 说明文档（可选）
   ```

2. **模型类型识别**
   - 文件名包含 `paraformer`：使用 `model_type: paraformer`
   - 文件名包含 `sense-voice`：使用 `model_type: sense_voice`
   - 不确定时查看 README 或模型说明

3. **验证方法**
   ```bash
   # 下载后解压查看文件
   tar -tf downloaded-model.tar.bz2 | grep -E "(model\.int8\.onnx|tokens\.txt)"
   
   # 如果两个文件都存在，模型可用
   ```

### 搜索技巧

1. **GitHub 搜索关键词**
   - `sherpa-onnx-paraformer-zh`（中文 Paraformer）
   - `sherpa-onnx-sense-voice`（多语言 SenseVoice）

2. **ModelScope 搜索**
   - 搜索 `sherpa-onnx`
   - 筛选作者 `pengzhendong`（经过验证的模型）

3. **识别不可用的模型**
   - ❌ 只有 `.pt` 或 `.pth` 文件（PyTorch 格式）
   - ❌ 只有 `.bin` 文件（未转换的二进制）
   - ❌ 缺少 `tokens.txt`
   - ✅ 包含 `model.int8.onnx` 和 `tokens.txt`

## 模型下载

### 推荐的 Paraformer 模型

#### 1. Paraformer-zh-small（推荐）
- **大小**：约 78MB
- **适用场景**：低性能设备，纯中文场景
- **下载地址**：
  ```bash
  # GitHub 官方下载
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2
  
  # 国内镜像加速
  wget https://ghproxy.com/https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2
  
  # 解压
  mkdir -p models
  tar -xf sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2 -C models/
  ```

#### 2. Paraformer-zh（标准版）
- **大小**：约 230MB
- **适用场景**：性能充足时，追求更高准确度
- **下载地址**：
  ```bash
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
  ```

### 其他可用的 Paraformer 模型

在 [Sherpa-ONNX Releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) 页面可找到：
- **中文版本**：搜索 "paraformer-zh"
- **英文版本**：搜索 "paraformer-en"
- **多语言版本**：搜索 "paraformer-trilingual"（中英日）

## 配置方法

在 `config.yaml` 中配置对应的 ASR 模块：

### 1. 在 config.yaml 中添加 Paraformer 配置

```yaml
ASR:
  SherpaParaformerASR:
    type: sherpa_onnx_local
    model_dir: models/sherpa-onnx-paraformer-zh-small-2024-03-09
    output_dir: tmp/
    model_type: paraformer  # 必须指定为 paraformer
```

### 2. 在 .config.yaml 中启用

```yaml
selected_module:
  ASR: SherpaParaformerASR  # 使用 Paraformer 模型
```

## Paraformer 不同版本对比

| 版本                | 模型大小 | 内存占用 | 推理速度 | 准确度 | 推荐场景   |
| ------------------- | -------- | -------- | -------- | ------ | ---------- |
| paraformer-zh-small | ~78MB    | ~200MB   | 快       | 良好   | 低性能设备 |
| paraformer-zh       | ~230MB   | ~500MB   | 中等     | 优秀   | 标准设备   |
| paraformer-zh-large | ~600MB   | ~1.2GB   | 较慢     | 最佳   | 高性能设备 |

## 实测性能数据

### RK3566 设备实测（2GB 内存）

使用 **paraformer-zh-small** 模型的实际表现：

| 测试项目 | 数据 |
|---------|------|
| 模型加载时间 | ~3秒 |
| 30字中文识别 | **~0.5秒** |
| 60字中文识别 | ~0.8秒 |
| 内存占用（空闲） | ~180MB |
| 内存占用（识别中） | ~220MB |
| CPU 占用（识别时） | **四核瞬时跑满** |

**测试环境**：
- 设备：RK3566 开发板
- 内存：2GB DDR4
- 系统：Debian 11
- Python：3.9

**性能优势**：
- ✅ 0.5秒完成30字识别，响应迅速
- ✅ 内存占用低，2GB 设备运行流畅
- ✅ 虽然识别瞬间 CPU 跑满，但持续时间短，不影响整体体验

**性能特点**：
- 识别过程中 CPU 四核会瞬时跑满（纯 CPU 推理）
- 由于识别速度快（0.5秒），CPU 高负载时间很短
- 对比 SenseVoice（894MB）在同设备上需要 2-3 秒，Paraformer-small（78MB）提升了 **4-6 倍**的识别速度

## 常见问题

### 1. 模型文件缺失错误

如果看到 "模型文件下载失败" 错误，请手动下载模型文件并确保以下文件存在：
- `model.int8.onnx` - 量化后的 ONNX 模型文件
- `tokens.txt` - 词汇表文件

### 2. RK3566 NPU 加速

目前 Sherpa-ONNX 暂不支持 RK3566 NPU 加速（正在努力攻克中），但 INT8 量化模型已经能在 CPU 上高效运行。

### 3. 模型选择建议

- **RK3566 等嵌入式设备**：使用 paraformer-zh-small
- **树莓派、迷你主机**：使用 paraformer-zh
- **性能充足的设备**：使用 paraformer-zh-large
- **需要极致性能**：考虑使用在线 ASR 服务（如豆包、阿里云）

### 4. 下载失败处理

**GitHub 下载慢或失败**：
- 使用代理工具
- 使用 GitHub 加速服务
- 从 ModelScope 下载同名模型

**ModelScope 下载失败**：
- 检查网络连接
- 使用手动下载方式
- 尝试其他模型源

### 5. 模型版本更新

定期检查新版本：
```bash
# 查看最新的 Paraformer 模型
curl -s https://api.github.com/repos/k2-fsa/sherpa-onnx/releases | grep paraformer-zh
```

## 实战示例

### 快速测试模型是否可用

```python
#!/usr/bin/env python3
# test_model.py - 测试下载的模型是否可用

import os
import sys

def check_model(model_dir):
    """检查模型文件是否完整"""
    required_files = ['model.int8.onnx', 'tokens.txt']
    missing = []
    
    for file in required_files:
        path = os.path.join(model_dir, file)
        if not os.path.exists(path):
            missing.append(file)
        else:
            size = os.path.getsize(path) / 1024 / 1024  # MB
            print(f"✓ {file}: {size:.1f} MB")
    
    if missing:
        print(f"✗ 缺少文件: {', '.join(missing)}")
        return False
    
    # 检查模型类型
    if 'paraformer' in model_dir.lower():
        print("→ 模型类型: paraformer")
    elif 'sense-voice' in model_dir.lower():
        print("→ 模型类型: sense_voice")
    else:
        print("→ 模型类型: 未知（默认使用 sense_voice）")
    
    return True

if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/sherpa-onnx-paraformer-zh-small-2024-03-09"
    if check_model(model_dir):
        print(f"\n✅ 模型 {model_dir} 可以使用!")
    else:
        print(f"\n❌ 模型 {model_dir} 不完整，请重新下载!")
```

## 参考链接

- [Sherpa-ONNX 官方仓库](https://github.com/k2-fsa/sherpa-onnx)
- [Sherpa-ONNX 模型下载](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)
- [Paraformer 论文](https://arxiv.org/abs/2206.08317)
- [k2-fsa 组织主页](https://github.com/k2-fsa)