<template>
  <el-dialog :title="$t('mcpToolCall.title')" :visible="visible" @close="handleClose" width="80%">
    <!-- 右上角操作按钮 -->
    <div slot="title" class="dialog-title-wrapper">
      <span class="dialog-title-text">{{ $t('mcpToolCall.title') }}</span>
    </div>

    <div class="dialog-content">
      <div class="main-layout">
        <!-- 左侧工具列表 -->
        <div class="left-panel">
          <div class="tool-list-section">
            <div class="section-header">
              <h3 class="section-title">
                <i class="el-icon-menu section-icon"></i>
                {{ $t('mcpToolCall.chooseFunction') }}
              </h3>
              <div class="tool-search">
                <el-input v-model="toolSearchKeyword" :placeholder="$t('mcpToolCall.searchFunction')" clearable />
              </div>
            </div>
            <div class="tool-list">
              <el-radio-group v-model="selectedToolName" class="tool-radio-group">
                <el-radio v-for="tool in filteredToolList" :key="tool.name" :label="tool.name" class="tool-radio">
                  <div class="tool-item">
                    <div class="tool-main-info">
                      <span class="tool-display-name">{{ getToolDisplayName(tool.name) }}</span>
                      <span class="tool-category">
                        {{ getToolCategory(tool.name) }}
                      </span>
                    </div>
                    <div class="tool-description">{{ getSimpleDescription(tool.description) }}</div>
                  </div>
                </el-radio>
              </el-radio-group>
            </div>

            <div v-if="filteredToolList.length === 0" class="no-results">
              <i class="el-icon-search no-results-icon"></i>
              <div class="no-results-text">{{ $t('mcpToolCall.noResults') }}</div>
            </div>
          </div>
        </div>

        <!-- 右侧面板 - 分为上下两部分 -->
        <div class="right-panel">
          <!-- 上部分：参数设置 -->
          <div v-if="selectedTool" class="params-section">
            <h3 class="params-title">
              <i class="el-icon-setting params-icon"></i>
              {{ $t('mcpToolCall.settings') }}
            </h3>

            <div class="params-help">
              {{ getToolHelpText(selectedTool.name) }}
            </div>

            <el-form :model="toolParams" :rules="toolParamsRules" ref="toolParamsForm" label-width="120px">
              <div v-for="(property, key) in selectedTool.inputSchema.properties" :key="key">
                <el-form-item :label="formatPropertyLabel(key, property)" :prop="key">
                  <template
                    v-if="property.type === 'integer' && property.minimum !== undefined && property.maximum !== undefined">
                    <el-input-number v-model="toolParams[key]" :min="property.minimum" :max="property.maximum"
                      :placeholder="$t('mcpToolCall.inputPlaceholder', { label: formatPropertyLabel(key, property) })"
                      style="width: 100%" />
                    <div class="param-range-hint">{{ $t('mcpToolCall.valueRange', {
                      min: property.minimum, max:
                        property.maximum
                    }) }}</div>
                  </template>
                  <template v-else-if="property.type === 'string' && (property.enum || key === 'theme')">
                    <el-select v-model="toolParams[key]"
                      :placeholder="$t('mcpToolCall.selectPlaceholder', { label: formatPropertyLabel(key, property) })"
                      style="width: 100%" clearable @change="handleThemeChange">
                      <template v-if="key === 'theme'">
                        <el-option v-for="option in themeOptions" :key="option.value" :label="option.label"
                          :value="option.value"></el-option>
                      </template>
                      <template v-else>
                        <el-option v-for="enumValue in property.enum" :key="enumValue" :label="enumValue"
                          :value="enumValue"></el-option>
                      </template>
                    </el-select>
                  </template>
                  <el-input v-else v-model="toolParams[key]"
                    :placeholder="$t('mcpToolCall.inputPlaceholder', { label: formatPropertyLabel(key, property) })"
                    :type="property.type === 'integer' ? 'number' : 'text'" style="width: 100%" />
                </el-form-item>
              </div>
            </el-form>
          </div>

          <div v-else class="no-selection">
            <i class="el-icon-info no-selection-icon"></i>
            <div class="no-selection-text">{{ $t('mcpToolCall.pleaseSelect') }}</div>
          </div>

          <!-- 下部分：执行结果 -->
          <div v-if="selectedTool" class="result-section">
            <h3 class="result-title">
              <i class="el-icon-document result-icon"></i>
              {{ $t('mcpToolCall.executionResult') }}
            </h3>

            <div v-if="executionResult" class="result-content">
              <pre class="result-text">{{ formattedExecutionResult }}</pre>
            </div>
            <div v-else class="no-result">
              <i class="el-icon-info no-result-icon"></i>
              <div class="no-result-text">{{ $t('mcpToolCall.noResultYet') }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 底部按钮区域 -->
    <div class="dialog-footer">
      <div class="dialog-btn cancel-btn" @click="cancel" style="flex: none; width: 100px;">{{ $t('mcpToolCall.cancel')
      }}
      </div>
      <el-button type="primary" @click="executeTool" size="small" class="execute-btn" style="margin-left: 10px;">
        <i class="el-icon-check"></i>
        {{ $t('mcpToolCall.execute') }}
      </el-button>
    </div>
  </el-dialog>
</template>

<script>
export default {
  name: 'McpToolCallDialog',
  props: {
    visible: { type: Boolean, required: true },
    deviceId: { type: String, required: true }
  },
  data() {
    return {
      toolList: [],
      selectedToolName: '',
      toolParams: {},
      toolParamsRules: {},
      toolSearchKeyword: '',
      executionResult: null,
      themeOptions: [] // 先初始化为空数组
    }
  },
  created() {
    // 在created钩子中初始化themeOptions，此时this.$t已经可用
    this.themeOptions = [
      { label: this.$t('mcpToolCall.lightTheme'), value: 'light' },
      { label: this.$t('mcpToolCall.darkTheme'), value: 'dark' }
    ];
  },
  computed: {
    selectedTool() {
      return this.toolList.find(tool => tool.name === this.selectedToolName);
    },
    filteredToolList() {
      if (!this.toolSearchKeyword) return this.toolList;
      const keyword = this.toolSearchKeyword.toLowerCase();
      return this.toolList.filter(tool =>
        tool.name.toLowerCase().includes(keyword) ||
        tool.description.toLowerCase().includes(keyword)
      );
    },
    formattedExecutionResult() {
      if (!this.executionResult) return '';
      return JSON.stringify(this.executionResult, null, 2);
    }
  },
  watch: {
    visible(newVal) {
      if (newVal) {
        this.initTools();
      }
    },
    selectedToolName(newVal) {
      if (newVal) {
        this.initToolParams();
        this.generateToolParamsRules();
      } else {
        this.executionResult = null;
      }
    }
  },
  methods: {
    // 添加handleThemeChange方法强制更新视图
    handleThemeChange() {
      this.$nextTick(() => {
        // 强制重新渲染组件
        this.$forceUpdate();
      });
    },
    initTools() {
      // 从固定的mcp工具列表中获取数据
      const toolsData = {
        "session_id": "", "type": "mcp", "payload": {
          "jsonrpc": "2.0", "id": 2, "result": {
            "tools": [{
              "name": "self.get_device_status",
              "description": "获取设备的实时信息，包括当前音频扬声器、屏幕、电池、网络等状态。\n用途：\n1. 回答关于当前设备状态的问题（例如：当前音频扬声器的音量是多少？）\n2. 作为控制设备的第一步（例如：调高/调低音频扬声器的音量等）",
              "inputSchema": { "type": "object", "properties": {} }
            }, {
              "name": "self.audio_speaker.set_volume",
              "description": "设置音频扬声器的音量。如果当前音量未知，必须先调用`self.get_device_status`工具，然后再调用此工具。",
              "inputSchema": {
                "type": "object",
                "properties": {
                  "volume": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100
                  }
                },
                "required": ["volume"]
              }
            }, {
              "name": "self.screen.set_brightness",
              "description": "设置屏幕的亮度。",
              "inputSchema": {
                "type": "object",
                "properties": {
                  "brightness": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100
                  }
                },
                "required": ["brightness"]
              }
            }, {
              "name": "self.screen.set_theme",
              "description": "设置屏幕的主题。主题可以是'light'（浅色）或'dark'（深色）。",
              "inputSchema": {
                "type": "object",
                "properties": { "theme": { "type": "string" } },
                "required": ["theme"]
              }
            }]
          }
        }
      };

      this.toolList = toolsData.payload.result.tools;
      // 默认选择第一个工具
      if (this.toolList.length > 0) {
        this.selectedToolName = this.toolList[0].name;
      }
    },

    initToolParams() {
      this.toolParams = {};
      if (this.selectedTool && this.selectedTool.inputSchema && this.selectedTool.inputSchema.properties) {
        Object.keys(this.selectedTool.inputSchema.properties).forEach(key => {
          // 根据工具名称和参数名设置默认值
          if (this.selectedTool.name === 'self.audio_speaker.set_volume' && key === 'volume') {
            this.toolParams[key] = 100; // 音量默认值设为100
          } else if (this.selectedTool.name === 'self.screen.set_brightness' && key === 'brightness') {
            this.toolParams[key] = 100; // 亮度默认值设为100
          } else if (this.selectedTool.name === 'self.screen.set_theme' && key === 'theme') {
            this.toolParams[key] = 'light'; // 主题默认值设为light
          } else {
            this.toolParams[key] = '';
          }
        });
      }
      this.executionResult = null;
    },

    generateToolParamsRules() {
      this.toolParamsRules = {};
      if (this.selectedTool && this.selectedTool.inputSchema && this.selectedTool.inputSchema.properties) {
        const requiredFields = this.selectedTool.inputSchema.required || [];

        Object.keys(this.selectedTool.inputSchema.properties).forEach(key => {
          const property = this.selectedTool.inputSchema.properties[key];
          const rules = [];

          if (requiredFields.includes(key)) {
            rules.push({ required: true, message: this.$t('mcpToolCall.requiredField', { field: this.formatPropertyLabel(key, property) }), trigger: 'blur' });
          }

          if (property.type === 'integer') {
            if (property.minimum !== undefined) {
              rules.push({
                validator: (rule, value, callback) => {
                  if (value < property.minimum) {
                    callback(new Error(this.$t('mcpToolCall.minValue', { value: property.minimum })));
                  } else {
                    callback();
                  }
                },
                trigger: 'blur'
              });
            }
            if (property.maximum !== undefined) {
              rules.push({
                validator: (rule, value, callback) => {
                  if (value > property.maximum) {
                    callback(new Error(this.$t('mcpToolCall.maxValue', { value: property.maximum })));
                  } else {
                    callback();
                  }
                },
                trigger: 'blur'
              });
            }
          }

          this.toolParamsRules[key] = rules;
        });
      }
    },

    formatPropertyLabel(key, property) {
      // 将属性名转换为更友好的中文标签
      const labelMap = {
        'volume': '音量',
        'brightness': '亮度',
        'theme': '主题'
      };
      return labelMap[key] || key;
    },

    // 获取工具的显示名称（简化版）
    getToolDisplayName(toolName) {
      const nameMap = {
        'self.get_device_status': '查看设备状态',
        'self.audio_speaker.set_volume': '设置音量',
        'self.screen.set_brightness': '设置亮度',
        'self.screen.set_theme': '设置主题'
      };
      return nameMap[toolName] || toolName;
    },

    // 获取工具分类
    getToolCategory(toolName) {
      if (toolName.includes('audio_speaker')) return '音频';
      if (toolName.includes('screen')) return '显示';
      return '设备信息';
    },

    // 获取简化的工具描述
    getSimpleDescription(originalDesc) {
      // 移除代码格式和复杂说明，保留核心功能描述
      return originalDesc.split('\n')[0].replace(/`/g, '');
    },

    // 获取工具帮助文本
    getToolHelpText(toolName) {
      const helpMap = {
        'self.get_device_status': '查看设备的当前运行状态，包括音量、屏幕、电池等信息。',
        'self.audio_speaker.set_volume': '调整设备的音量大小，请输入0-100之间的数值。',
        'self.screen.set_brightness': '调整设备屏幕的亮度，请输入0-100之间的数值。',
        'self.screen.set_theme': '切换设备屏幕的显示主题，可以选择浅色或深色模式。'
      };
      return helpMap[toolName] || '';
    },

    executeTool() {
      if (!this.selectedTool) {
        this.$message.warning(this.$t('mcpToolCall.selectTool'));
        return;
      }

      // 验证必填参数
      const requiredFields = this.selectedTool.inputSchema.required || [];
      for (const field of requiredFields) {
        if (this.toolParams[field] === undefined || this.toolParams[field] === null || this.toolParams[field] === '') {
          this.$message.warning(this.$t('mcpToolCall.requiredField', { field: this.formatPropertyLabel(field, this.selectedTool.inputSchema.properties[field]) }));
          return;
        }
      }

      // 构建MCP执行字符串
      const mcpExecuteString = {
        "type": "mcp",
        "payload": {
          "jsonrpc": "2.0",
          "id": 1,
          "method": "tools/call",
          "params": {
            "name": this.selectedToolName,
            "arguments": this.toolParams
          }
        }
      };

      // 存储执行结果
      this.executionResult = mcpExecuteString;
    },

    cancel() {
      this.closeDialog();
    },

    handleClose() {
      this.closeDialog();
    },

    closeDialog() {
      this.$emit('update:visible', false);
      this.selectedToolName = '';
      this.toolParams = {};
      this.toolParamsRules = {};
      this.toolSearchKeyword = '';
      this.executionResult = null;
    }
  }
}
</script>

<style scoped>
.dialog-content {
  padding: 0;
}

/* 对话框标题区域 */
.dialog-title-wrapper {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.dialog-title-text {
  font-size: 18px;
  font-weight: 600;
  color: #303133;
}

.dialog-top-actions {
  display: flex;
  gap: 10px;
}

.execute-btn {
  border-radius: 6px;
  padding: 8px 16px;
  font-size: 14px;
  font-weight: 500;
}

/* 主布局 */
.main-layout {
  display: flex;
  gap: 20px;
  height: calc(100vh - 260px);
  min-height: 400px;
}

/* 左侧面板 */
.left-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  border: 1px solid #e4e7ed;
  border-radius: 12px;
  overflow: hidden;
}

.tool-list-section {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.section-header {
  padding: 0px 20px 20px 20px;
  border-bottom: 1px solid #e4e7ed;
  background: #fafafa;
}

.section-title {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.section-icon {
  font-size: 18px;
  color: #5778ff;
}

.tool-search {
  width: 100%;
}

::v-deep .tool-search .el-input__wrapper {
  border-radius: 8px;
  transition: all 0.3s ease;
}

::v-deep .tool-search .el-input__wrapper:hover {
  border-color: #c0c4cc;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06);
}

::v-deep .tool-search .el-input__wrapper.is-focus {
  border-color: #5778ff;
  box-shadow: 0 0 0 2px rgba(87, 120, 255, 0.2);
}

/* 工具列表 */
.tool-list {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
}

::v-deep .tool-list::-webkit-scrollbar {
  width: 6px;
}

::v-deep .tool-list::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

::v-deep .tool-list::-webkit-scrollbar-thumb {
  background: #c0c4cc;
  border-radius: 3px;
}

::v-deep .tool-list::-webkit-scrollbar-thumb:hover {
  background: #909399;
}

.tool-radio-group {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

/* 修复单选按钮对齐问题 */
::v-deep .el-radio {
  display: flex !important;
  align-items: flex-start !important;
}

::v-deep .el-radio__input {
  margin-top: 6px;
  margin-right: 10px;
  flex-shrink: 0;
}

::v-deep .el-radio__label {
  flex: 1;
  padding: 0 !important;
}

.tool-radio {
  background-color: #f8f9fa;
  border-radius: 12px;
  padding: 16px;
  cursor: pointer;
  transition: all 0.3s ease;
  border: 2px solid transparent;
}

.tool-radio:hover {
  background-color: #f0f2f5;
  border-color: #e4e7ed;
  transform: translateX(2px);
}

::v-deep .tool-radio.is-checked {
  background-color: #e6f7ff;
  border-color: #5778ff;
}

::v-deep .el-radio__input.is-checked .el-radio__inner {
  border-color: #5778ff;
  background: #5778ff;
}

::v-deep .el-radio__input.is-checked+.el-radio__label {
  color: #5778ff;
}

.tool-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.tool-main-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

.tool-display-name {
  font-weight: 600;
  font-size: 14px;
  color: #303133;
  flex: 1;
  text-align: left;
}

.tool-category {
  background: #ecf5ff;
  color: #409eff;
  font-size: 12px;
  padding: 4px 10px;
  border-radius: 16px;
  font-weight: 500;
}

.tool-description {
  font-size: 13px;
  color: #606266;
  line-height: 1.6;
  white-space: pre-wrap;
  opacity: 0.9;
  text-align: left;
}

/* 右侧面板 - 分为上下两部分 */
.right-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  border: 1px solid #e4e7ed;
  border-radius: 12px;
  overflow: hidden;
}

/* 参数设置区域 */
.params-section {
  padding: 0px 20px 20px 20px;
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.params-title {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 20px;
}

.params-icon {
  font-size: 18px;
  color: #5778ff;
}

.params-help {
  background: #f8f9ff;
  border: 1px solid #ebefff;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 20px;
  font-size: 13px;
  color: #606266;
  line-height: 1.5;
}

/* 表单样式 */
::v-deep .el-form-item {
  margin-bottom: 20px;
}

::v-deep .el-form-item__label {
  font-size: 14px;
  color: #606266;
  font-weight: 500;
}

::v-deep .el-form-item__content {
  font-size: 14px;
}

.param-range-hint {
  font-size: 12px;
  color: #909399;
  margin-top: 6px;
}

::v-deep .el-input__wrapper {
  border-radius: 8px;
  transition: all 0.3s ease;
}

::v-deep .el-input__wrapper:hover {
  border-color: #c0c4cc;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06);
}

::v-deep .el-input__wrapper.is-focus {
  border-color: #5778ff;
  box-shadow: 0 0 0 2px rgba(87, 120, 255, 0.2);
}

::v-deep .el-select .el-input__wrapper {
  border-radius: 8px;
}

::v-deep .el-input-number {
  border-radius: 8px;
  overflow: hidden;
}

/* 执行结果区域 */
.result-section {
  padding: 20px;
  background: #fafafa;
  border-top: 1px solid #e4e7ed;
  max-height: 200px;
  display: flex;
  flex-direction: column;
  min-height: 150px;
}

.result-title {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.result-icon {
  font-size: 18px;
  color: #5778ff;
}

.result-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: white;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  padding: 12px;
  position: relative;
}

.result-text {
  flex: 1;
  margin: 0;
  padding: 8px;
  background: #f8f9fa;
  border: 1px solid #e4e7ed;
  border-radius: 4px;
  font-size: 12px;
  font-family: 'Courier New', monospace;
  white-space: pre-wrap;
  word-wrap: break-word;
  overflow-y: auto;
  max-height: 120px;
  text-align: left;
}

.copy-btn {
  align-self: flex-end;
  margin-top: 8px;
  padding: 4px 12px;
  font-size: 12px;
}

.no-selection,
.no-results,
.no-result {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #909399;
  text-align: center;
  padding: 40px 20px;
}

.no-selection-icon,
.no-results-icon,
.no-result-icon {
  font-size: 48px;
  color: #c0c4cc;
  margin-bottom: 16px;
}

.no-selection-text,
.no-results-text,
.no-result-text {
  font-size: 14px;
}

/* 底部按钮区域 */
.dialog-footer {
  display: flex;
  justify-content: center;
  margin: 20px 0;
  padding-top: 0;
}

.dialog-btn {
  cursor: pointer;
  border-radius: 8px;
  height: 40px;
  font-weight: 500;
  font-size: 14px;
  line-height: 40px;
  text-align: center;
  transition: all 0.3s ease;
}

.cancel-btn {
  background: #f5f7fa;
  border: 1px solid #e4e7ed;
  color: #606266;
}

.cancel-btn:hover {
  background: #e9ecef;
  border-color: #dcdfe6;
  color: #409eff;
}

/* 对话框整体样式 */
::v-deep .el-dialog {
  border-radius: 16px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.12);
  overflow: hidden;
  animation: dialogFadeIn 0.3s ease-out;
  max-width: 1200px;
  margin-top: 3% !important;
}

@keyframes dialogFadeIn {
  from {
    opacity: 0;
    transform: translateY(-20px);
  }

  to {
    opacity: 1;
    transform: translateY(0);
  }
}

::v-deep .el-dialog__header {
  padding: 24px 24px 0;
}

::v-deep .el-dialog__body {
  padding: 20px 24px 0;
  max-height: 80vh;
  overflow-y: auto;
}

/* 响应式调整 */
@media (max-width: 1200px) {
  ::v-deep .el-dialog {
    width: 95% !important;
  }

  .main-layout {
    flex-direction: column;
    height: auto;
    min-height: 0;
  }

  .left-panel,
  .right-panel {
    max-height: 400px;
  }
}
</style>