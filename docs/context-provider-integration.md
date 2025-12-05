# 数据上下文填充功能使用指南

## 概述

数据上下文填充功能（Context Data Provider）允许小智在唤醒那一刻，获取外部系统的数据，并将其动态注入到大模型的系统提示词（System Prompt）中。
让其做到唤醒时感知世界某个事物的状态。

通过这个功能，在小智唤醒的一刹那，“感知”到：
- 智能家居的传感器状态（温度、湿度、灯光状态等）
- 业务系统的实时数据（服务器负载、健康数据、股票信息等）
- 任何可以通过 HTTP API 获取的文本信息

**注意**：该功能只是方便小智在唤醒的时候感知事物的状态，而如果想要小智唤醒后实时获取事物的状态，建议在此功能上再结合MCP工具的调用。

## 工作原理

1. **配置源**：用户配置一个或多个 HTTP API 地址。
2. **触发请求**：当系统构建 Prompt 时，如果发现模板中包含 `{{ dynamic_context }}` 占位符，会请求所有配置的 API。
3. **自动注入**：系统会自动将 API 返回的数据格式化为 Markdown 列表，替换 `{{ dynamic_context }}` 占位符。

## 接口规范

为了让小智正确解析数据，您的 API 需要满足以下规范：

- **请求方式**：`GET`
- **请求头**：系统会自动添加 `device_id` 字段到 Request Header。
- **响应格式**：必须返回 JSON 格式，且包含 `code` 和 `data` 字段。

### 响应示例

**情况 1：返回键值对**
```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "客厅温度": "26℃",
    "客厅湿度": "45%",
    "大门状态": "已关闭"
  }
}
```
*注入效果：*
```markdown
<context>
- **客厅温度：** 26℃
- **客厅湿度：** 45%
- **大门状态：** 已关闭
</context>
```

**情况 2：返回列表**
```json
{
  "code": 0,
  "data": [
    "您有10个待办事项",
    "当前汽车的行驶速度是100km每小时"
  ]
}
```
*注入效果：*
```markdown
<context>
- 您有10个待办事项
- 当前汽车的行驶速度是100km每小时
</context>
```

## 配置指南

### 方式 1：智控台配置（全模块部署）

1. 登录智控台，进入**角色配置**页面。
2. 找到**上下文源**配置项（点击“编辑源”按钮）。
3. 点击**添加**，输入您的 API 地址。
4. 如果 API 需要鉴权，可以在**请求头**部分添加 `Authorization` 或其他 Header。
5. 保存配置。

### 方式 2：配置文件配置（单模块部署）

编辑 `xiaozhi-server/data/.config.yaml` 文件，添加 `context_providers` 配置段：

```yaml
# 上下文源配置
context_providers:
  - url: "http://api.example.com/data"
    headers:
      Authorization: "Bearer your-token"
  - url: "http://another-api.com/data"
```

## 启用功能

默认情况下，系统的提示词模板文件（`data/.agent-base-prompt.txt`）中已经预置了 `{{ dynamic_context }}` 占位符，您无需手动添加。

**示例：**

```markdown
<context>
【重要！以下信息已实时提供，无需调用工具查询，请直接使用：】
- **设备ID：** {{device_id}}
- **当前时间：** {{current_time}}
...
{{ dynamic_context }}
</context>
```

**注意**：如果您不需要使用此功能，可以选择**不配置任何上下文源**，也可以从提示词模板文件中**删除** `{{ dynamic_context }}` 占位符。
