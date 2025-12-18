# 单模块部署固件OTA自动升级配置指南

本教程将指导你如何在**单模块部署**场景下配置固件OTA自动升级功能，实现设备固件的自动更新。

## 功能介绍

在单模块部署中，xiaozhi-server内置了OTA固件管理功能，可以自动检测设备版本并下发升级固件。系统会根据设备型号和当前版本，自动匹配并推送最新的固件版本。

## 前提条件

- 你已经成功进行**单模块部署**并运行xiaozhi-server
- 设备能够正常连接到服务器

## 第一步 准备固件文件

### 1. 创建固件存放目录

固件文件需要放在`data/bin/`目录下。如果该目录不存在，请手动创建：

```bash
mkdir -p data/bin
```

### 2. 固件文件命名规则

固件文件必须遵循以下命名格式：

```
{设备型号}_{版本号}.bin
```

**命名规则说明：**
- `设备型号`：设备的型号名称，例如 `lichuang-dev`、`bread-compact-wifi` 等
- `版本号`：固件版本号，必须以数字开头，支持数字、字母、点号、下划线和短横线，例如 `1.6.6`、`2.0.0` 等
- 文件扩展名必须是 `.bin`

**命名示例：**
```
bread-compact-wifi_1.6.6.bin
lichuang-dev_2.0.0.bin
```

### 3. 放置固件文件

将准备好的固件文件（.bin文件）复制到`data/bin/`目录下：

```bash
cp your_firmware.bin data/bin/设备型号_版本号.bin
```

例如：
```bash
cp xiaozhi_firmware.bin data/bin/esp32s3_1.6.6.bin
```

## 第二步 配置公网访问地址（仅公网部署需要）

**注意：此步骤仅适用于单模块公网部署的场景。**

如果你的xiaozhi-server是公网部署（使用公网IP或域名），**必须**配置`server.vision_explain`参数，因为OTA固件下载地址会使用该配置的域名和端口。

如果你是局域网部署，可以跳过此步骤。

### 为什么要配置这个参数？

在单模块部署中，系统生成固件下载地址时，会使用`vision_explain`配置的域名和端口作为基础地址。如果不配置或配置错误，设备将无法访问固件下载地址。

### 配置方法

打开`data/.config.yaml`文件，找到`server`配置段，设置`vision_explain`参数：

```yaml
server:
  vision_explain: http://你的域名或IP:端口号/mcp/vision/explain
```

**配置示例：**

局域网部署（默认）：
```yaml
server:
  vision_explain: http://192.168.1.100:8003/mcp/vision/explain
```

公网域名部署：
```yaml
server:
  vision_explain: http://yourdomain.com:8003/mcp/vision/explain
```

公网IP部署：
```yaml
server:
  vision_explain: http://111.111.111.111:8003/mcp/vision/explain
```

使用HTTPS（推荐公网部署使用）：
```yaml
server:
  vision_explain: https://yourdomain.com:8003/mcp/vision/explain
```

### 注意事项

- 域名或IP必须是设备能够访问的地址
- 如果使用Docker部署，不能使用Docker内部地址（如127.0.0.1或localhost）
- 端口号默认是8003，如果你修改了`server.http_port`配置，需要同步修改这里的端口号

## 第三步 启动或重启服务

### 源码运行

```bash
python app.py
```

### Docker运行

```bash
docker restart xiaozhi-esp32-server
```

### 验证服务启动

启动后，查看日志输出，应该能看到类似以下内容：

```
2025-12-18 **** - OTA接口是           http://192.168.1.100:8003/xiaozhi/ota/
2025-12-18 **** - 视觉分析接口是        http://192.168.1.100:8003/mcp/vision/explain
```

使用浏览器访问OTA接口地址，如果显示以下内容说明服务正常：

```
OTA接口运行正常，向设备发送的websocket地址是：ws://xxx.xxx.xxx.xxx:8000/xiaozhi/v1/
```

## 第四步 设备自动检测升级

### 升级原理

当设备连接到服务器时（每次开机或定时检查），会自动发送OTA请求。服务器会：

1. 读取设备的型号和当前固件版本
2. 扫描`data/bin/`目录，查找匹配该型号的所有固件文件
3. 比较版本号，如果有更高版本，则返回固件下载地址
4. 设备收到下载地址后，会自动下载并安装新固件

### 版本比较规则

系统使用语义化版本比较方式，按数字段从左到右依次比较：

- `1.6.6` < `1.6.7`
- `1.6.9` < `1.7.0`
- `2.0.0` > `1.9.9`

### 查看升级日志

在xiaozhi-server的日志中，你可以看到OTA相关的日志输出：

```
[ota_handler] - OTA请求设备ID: AA:BB:CC:DD:EE:FF
[ota_handler] - 查找型号 esp32s3 的固件，找到 3 个候选
[ota_handler] - 为设备 AA:BB:CC:DD:EE:FF 下发固件 1.6.6 [如果地址前缀有误，请检查配置文件中的server.vision_explain]-> http://yourdomain.com:8003/xiaozhi/ota/download/esp32s3_1.6.6.bin
```

或者如果设备已是最新版本：

```
[ota_handler] - 设备 AA:BB:CC:DD:EE:FF 固件已是最新: 1.6.6
```

## 高级配置

### 固件缓存时间（可选）

系统会缓存`data/bin/`目录的扫描结果以提高性能。默认缓存时间为30秒。你可以在配置文件中调整：

```yaml
firmware_cache_ttl: 60  # 单位：秒，设置为60秒缓存时间
```

### 多版本固件管理

你可以同时放置多个版本的固件，系统会自动选择最新版本：

```
data/bin/
  ├── esp32s3_1.6.5.bin
  ├── esp32s3_1.6.6.bin
  ├── esp32s3_1.7.0-beta.bin
  └── xiaozhi-v2_2.0.0.bin
```

系统会为`esp32s3`型号的设备推送`1.7.0-beta`版本（最高版本）。

### 多型号固件管理

不同型号的设备会自动匹配对应型号的固件：

```
data/bin/
  ├── esp32s3_1.6.6.bin       # 仅供 esp32s3 型号设备使用
  ├── xiaozhi-v2_2.0.0.bin    # 仅供 xiaozhi-v2 型号设备使用
  └── default_1.0.0.bin       # 供未识别型号的设备使用
```

## 常见问题

### 1. 设备收不到固件更新

**可能原因和解决方法：**

- 检查固件文件命名是否符合规则：`{型号}_{版本号}.bin`
- 检查固件文件是否正确放置在`data/bin/`目录
- 检查设备型号是否与固件文件名中的型号匹配
- 检查固件版本号是否高于设备当前版本
- 查看服务器日志，确认OTA请求是否正常处理

### 2. 设备报告下载地址无法访问

**可能原因和解决方法：**

- 检查`server.vision_explain`配置的域名或IP是否正确
- 确认端口号配置正确（默认8003）
- 如果是公网部署，确保设备能够访问该公网地址
- 如果是Docker部署，确保不是使用了内部地址（127.0.0.1）
- 检查防火墙是否开放了对应端口

### 3. 如何确认设备当前版本

查看OTA请求日志，日志中会显示设备上报的版本号：

```
[ota_handler] - 设备 AA:BB:CC:DD:EE:FF 固件已是最新: 1.6.6
```

### 4. 固件文件放置后没有生效

系统有30秒的缓存时间（默认），可以：
- 等待30秒后再让设备发起OTA请求
- 重启xiaozhi-server服务
- 调整`firmware_cache_ttl`配置为更短的时间

### 5. 如何回滚到旧版本

系统只会推送更高版本的固件。如果需要回滚：
1. 删除或重命名`data/bin/`目录中高于目标版本的固件文件
2. 等待缓存过期或重启服务
3. 设备下次检查时会收到目标版本

## 安全说明

- 系统会验证固件文件路径，防止目录穿越攻击
- 固件下载接口只允许访问`data/bin/`目录下的`.bin`文件
- 建议在生产环境使用HTTPS协议传输固件

## 相关教程

如需了解更多，请参考以下教程：

1. [如何自己编译小智固件](./firmware-build.md)
2. [如何基于虾哥编译好的固件修改OTA地址](./firmware-setting.md)
3. [如何进行全模块部署](./Deployment_all.md)
