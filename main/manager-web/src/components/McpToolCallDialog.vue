<template>
  <el-dialog
    :title="$t('mcpToolCall.title')"
    :visible="visible"
    @close="handleClose"
    width="80%"
  >
    <!-- 右上角操作按钮 -->
    <div slot="title" class="dialog-title-wrapper">
      <span class="dialog-title-text">{{ $t("mcpToolCall.title") }}</span>
    </div>

    <div class="dialog-content">
      <div class="main-layout">
        <!-- 左侧工具列表 -->
        <div class="left-panel">
          <div class="tool-list-section">
            <div class="section-header">
              <h3 class="section-title">
                <i class="el-icon-menu section-icon"></i>
                {{ $t("mcpToolCall.chooseFunction") }}
              </h3>
              <div class="tool-search">
                <el-input
                  v-model="toolSearchKeyword"
                  :placeholder="$t('mcpToolCall.searchFunction')"
                  clearable
                />
              </div>
            </div>
            <div v-if="toolsLoading" class="tool-list-loading">
              <i class="el-icon-loading"></i>
              <div class="loading-text">正在获取工具列表...</div>
            </div>
            <div v-else class="tool-list">
              <el-radio-group v-model="selectedToolName" class="tool-radio-group">
                <el-radio
                  v-for="tool in filteredToolList"
                  :key="tool.name"
                  :label="tool.name"
                  class="tool-radio"
                >
                  <div class="tool-item">
                    <div class="tool-main-info">
                      <span class="tool-display-name">{{
                        getToolDisplayName(tool.name)
                      }}</span>
                      <span class="tool-category">
                        {{ getToolCategory(tool.name) }}
                      </span>
                    </div>
                    <div class="tool-description">
                      {{ getSimpleDescription(tool.description) }}
                    </div>
                  </div>
                </el-radio>
              </el-radio-group>
            </div>

            <div v-if="!toolsLoading && filteredToolList.length === 0" class="no-results">
              <i class="el-icon-search no-results-icon"></i>
              <div class="no-results-text">{{ $t("mcpToolCall.noResults") }}</div>
            </div>
          </div>
        </div>

        <!-- 右侧面板 - 分为上下两部分 -->
        <div class="right-panel">
          <!-- 上部分：参数设置 -->
          <div v-if="selectedTool" class="params-section">
            <h3 class="params-title">
              <i class="el-icon-setting params-icon"></i>
              {{ $t("mcpToolCall.settings") }}
            </h3>

            <div class="params-help">
              {{ getToolHelpText(selectedTool.name) }}
            </div>

            <el-form
              :model="toolParams"
              :rules="toolParamsRules"
              ref="toolParamsForm"
              label-width="120px"
            >
              <div
                v-for="(property, key) in selectedTool.inputSchema.properties"
                :key="key"
              >
                <el-form-item :label="formatPropertyLabel(key, property)" :prop="key">
                  <template
                    v-if="
                      property.type === 'integer' &&
                      property.minimum !== undefined &&
                      property.maximum !== undefined
                    "
                  >
                    <el-input-number
                      v-model="toolParams[key]"
                      :min="property.minimum"
                      :max="property.maximum"
                      style="width: 100%"
                    />
                  </template>
                  <template
                    v-else-if="
                      property.type === 'string' && (property.enum || key === 'theme')
                    "
                  >
                    <el-select
                      v-model="toolParams[key]"
                      style="width: 100%"
                      clearable
                      @change="handleThemeChange"
                    >
                      <template v-if="key === 'theme'">
                        <el-option
                          v-for="option in themeOptions"
                          :key="option.value"
                          :label="option.label"
                          :value="option.value"
                        ></el-option>
                      </template>
                      <template v-else>
                        <el-option
                          v-for="enumValue in property.enum"
                          :key="enumValue"
                          :label="enumValue"
                          :value="enumValue"
                        ></el-option>
                      </template>
                    </el-select>
                  </template>
                  <el-input
                    v-else
                    v-model="toolParams[key]"
                    :type="property.type === 'integer' ? 'number' : 'text'"
                    style="width: 100%"
                  />
                </el-form-item>
              </div>
            </el-form>
          </div>

          <div v-else class="no-selection">
            <i class="el-icon-info no-selection-icon"></i>
            <div class="no-selection-text">{{ $t("mcpToolCall.pleaseSelect") }}</div>
          </div>

          <!-- 下部分：执行结果 -->
          <div v-if="selectedTool" class="result-section">
            <h3 class="result-title">
              <i class="el-icon-document result-icon"></i>
              {{ $t("mcpToolCall.executionResult") }}
            </h3>

            <div v-if="executionResult" class="result-content">
              <!-- 表格展示模式 -->
              <div v-if="showResultAsTable" class="result-table">
                <el-table :data="tableData" border size="mini" style="width: 100%">
                  <el-table-column
                    prop="category"
                    label="组件"
                    width="120"
                  ></el-table-column>
                  <el-table-column
                    prop="property"
                    label="属性"
                    width="120"
                  ></el-table-column>
                  <el-table-column prop="value" label="值"></el-table-column>
                </el-table>
              </div>
              <!-- JSON展示模式 -->
              <pre v-else class="result-text">{{ formattedExecutionResult }}</pre>
            </div>
            <div v-else class="no-result">
              <i class="el-icon-info no-result-icon"></i>
              <div class="no-result-text">{{ $t("mcpToolCall.noResultYet") }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 底部按钮区域 -->
    <div class="dialog-footer">
      <div class="dialog-btn cancel-btn" @click="cancel" style="flex: none; width: 100px">
        {{ $t("mcpToolCall.cancel") }}
      </div>
      <el-button
        type="primary"
        @click="executeTool"
        size="small"
        class="execute-btn"
        style="margin-left: 10px"
      >
        <i class="el-icon-check"></i>
        {{ $t("mcpToolCall.execute") }}
      </el-button>
    </div>
  </el-dialog>
</template>

<script>
import Api from "@/apis/api";

export default {
  name: "McpToolCallDialog",
  props: {
    visible: { type: Boolean, required: true },
    deviceId: { type: String, required: true },
  },
  data() {
    return {
      toolList: [],
      selectedToolName: "",
      toolParams: {},
      toolParamsRules: {},
      toolSearchKeyword: "",
      executionResult: null,
      themeOptions: [], // 先初始化为空数组
      toolsLoading: false, // 工具列表加载状态
      showResultAsTable: false, // 是否以表格形式展示结果
      tableData: [], // 表格数据
    };
  },
  created() {
    // 在created钩子中初始化themeOptions，此时this.$t已经可用
    this.themeOptions = [
      { label: this.$t("mcpToolCall.lightTheme"), value: "light" },
      { label: this.$t("mcpToolCall.darkTheme"), value: "dark" },
    ];
  },
  computed: {
    selectedTool() {
      return this.toolList.find((tool) => tool.name === this.selectedToolName);
    },
    filteredToolList() {
      if (!this.toolSearchKeyword) return this.toolList;
      const keyword = this.toolSearchKeyword.toLowerCase();
      return this.toolList.filter(
        (tool) =>
          tool.name.toLowerCase().includes(keyword) ||
          tool.description.toLowerCase().includes(keyword)
      );
    },
    formattedExecutionResult() {
      if (!this.executionResult) return "";
      return JSON.stringify(this.executionResult, null, 2);
    },
  },
  watch: {
    visible(newVal) {
      if (newVal) {
        this.initTools();
      }
    },
    selectedToolName(newVal) {
      if (newVal) {
        // 延迟执行初始化，确保 selectedTool computed 已经更新
        this.$nextTick(() => {
          this.initToolParams();
          this.generateToolParamsRules();
          // 清空表单验证状态
          if (this.$refs.toolParamsForm) {
            this.$refs.toolParamsForm.clearValidate();
          }
        });
      } else {
        this.executionResult = null;
        this.toolParams = {};
        this.toolParamsRules = {};
      }
    },
  },
  methods: {
    // 添加handleThemeChange方法强制更新视图
    handleThemeChange() {
      this.$nextTick(() => {
        // 强制重新渲染组件
        this.$forceUpdate();
      });
    },

    // 检查是否为预定义工具
    isPredefinedTool(toolName) {
      const predefinedTools = [
        "self.get_device_status",
        "self.audio_speaker.set_volume",
        "self.screen.set_brightness",
        "self.screen.set_theme",
        "self.get_system_info",
        "self.reboot",
        "self.screen.get_info",
        "self.screen.snapshot",
      ];
      return predefinedTools.includes(toolName);
    },

    // 解析设备状态数据为表格格式
    parseDeviceStatusToTable(deviceData) {
      const tableData = [];

      if (deviceData.audio_speaker) {
        if (deviceData.audio_speaker.volume !== undefined) {
          tableData.push({
            category: "音频扬声器",
            property: "音量",
            value: deviceData.audio_speaker.volume + "%",
          });
        }
      }

      if (deviceData.screen) {
        if (deviceData.screen.brightness !== undefined) {
          tableData.push({
            category: "屏幕",
            property: "亮度",
            value: deviceData.screen.brightness + "%",
          });
        }
        if (deviceData.screen.theme !== undefined) {
          tableData.push({
            category: "屏幕",
            property: "主题",
            value: deviceData.screen.theme === "dark" ? "深色" : "浅色",
          });
        }
      }

      if (deviceData.network) {
        if (deviceData.network.type !== undefined) {
          tableData.push({
            category: "网络",
            property: "类型",
            value: deviceData.network.type.toUpperCase(),
          });
        }
        if (deviceData.network.ssid !== undefined) {
          tableData.push({
            category: "网络",
            property: "SSID",
            value: deviceData.network.ssid,
          });
        }
        if (deviceData.network.signal !== undefined) {
          const signalMap = {
            strong: "强",
            medium: "中",
            weak: "弱",
          };
          tableData.push({
            category: "网络",
            property: "信号强度",
            value: signalMap[deviceData.network.signal] || deviceData.network.signal,
          });
        }
      }

      return tableData;
    },

    // 解析其他工具结果为表格格式
    parseOtherResultToTable(toolName, result) {
      const tableData = [];

      if (toolName === "self.audio_speaker.set_volume") {
        tableData.push({
          category: "音频控制",
          property: "操作结果",
          value: result.success ? "设置成功" : "设置失败",
        });
      } else if (toolName === "self.screen.set_brightness") {
        tableData.push({
          category: "屏幕控制",
          property: "操作结果",
          value: result.success ? "亮度设置成功" : "亮度设置失败",
        });
      } else if (toolName === "self.screen.set_theme") {
        tableData.push({
          category: "屏幕控制",
          property: "操作结果",
          value: result.success ? "主题设置成功" : "主题设置失败",
        });
      } else if (toolName === "self.reboot") {
        tableData.push({
          category: "系统控制",
          property: "操作结果",
          value: result.success ? "重启指令已发送" : "重启失败",
        });
      } else if (toolName === "self.screen.get_info") {
        // 解析屏幕信息
        if (
          result.success &&
          result.data &&
          result.data.content &&
          result.data.content[0] &&
          result.data.content[0].text
        ) {
          try {
            const screenInfo = JSON.parse(result.data.content[0].text);
            if (screenInfo.width !== undefined) {
              tableData.push({
                category: "屏幕信息",
                property: "宽度",
                value: screenInfo.width + "像素",
              });
            }
            if (screenInfo.height !== undefined) {
              tableData.push({
                category: "屏幕信息",
                property: "高度",
                value: screenInfo.height + "像素",
              });
            }
            if (screenInfo.monochrome !== undefined) {
              tableData.push({
                category: "屏幕信息",
                property: "类型",
                value: screenInfo.monochrome ? "单色屏" : "彩色屏",
              });
            }
          } catch (parseError) {
            // 解析失败时显示原始信息
            tableData.push({
              category: "屏幕信息",
              property: "获取结果",
              value: result.success ? "获取成功，但解析失败" : "获取失败",
            });
          }
        } else {
          tableData.push({
            category: "屏幕信息",
            property: "获取结果",
            value: result.success ? "获取成功，但数据格式异常" : "获取失败",
          });
        }
      } else if (toolName === "self.get_system_info") {
        // 解析系统信息
        if (
          result.success &&
          result.data &&
          result.data.content &&
          result.data.content[0] &&
          result.data.content[0].text
        ) {
          try {
            const systemInfo = JSON.parse(result.data.content[0].text);

            // 基本信息
            if (systemInfo.chip_model_name) {
              tableData.push({
                category: "硬件信息",
                property: "芯片型号",
                value: systemInfo.chip_model_name.toUpperCase(),
              });
            }

            if (systemInfo.chip_info) {
              if (systemInfo.chip_info.cores) {
                tableData.push({
                  category: "硬件信息",
                  property: "CPU核心数",
                  value: systemInfo.chip_info.cores + "核",
                });
              }
              if (systemInfo.chip_info.revision) {
                tableData.push({
                  category: "硬件信息",
                  property: "芯片版本",
                  value: "Rev " + systemInfo.chip_info.revision,
                });
              }
            }

            if (systemInfo.flash_size) {
              tableData.push({
                category: "硬件信息",
                property: "Flash大小",
                value: (systemInfo.flash_size / 1024 / 1024).toFixed(0) + " MB",
              });
            }

            // 内存信息
            if (systemInfo.minimum_free_heap_size) {
              tableData.push({
                category: "内存信息",
                property: "最小可用堆",
                value:
                  (parseInt(systemInfo.minimum_free_heap_size) / 1024).toFixed(0) + " KB",
              });
            }

            // 应用信息
            if (systemInfo.application) {
              if (systemInfo.application.name) {
                tableData.push({
                  category: "应用信息",
                  property: "应用名称",
                  value: systemInfo.application.name,
                });
              }
              if (systemInfo.application.version) {
                tableData.push({
                  category: "应用信息",
                  property: "应用版本",
                  value: systemInfo.application.version,
                });
              }
              if (systemInfo.application.compile_time) {
                tableData.push({
                  category: "应用信息",
                  property: "编译时间",
                  value: systemInfo.application.compile_time,
                });
              }
              if (systemInfo.application.idf_version) {
                tableData.push({
                  category: "应用信息",
                  property: "IDF版本",
                  value: systemInfo.application.idf_version,
                });
              }
            }

            // 网络信息
            if (systemInfo.mac_address) {
              tableData.push({
                category: "网络信息",
                property: "MAC地址",
                value: systemInfo.mac_address,
              });
            }

            if (systemInfo.board) {
              if (systemInfo.board.ip) {
                tableData.push({
                  category: "网络信息",
                  property: "IP地址",
                  value: systemInfo.board.ip,
                });
              }
              if (systemInfo.board.ssid) {
                tableData.push({
                  category: "网络信息",
                  property: "WiFi名称",
                  value: systemInfo.board.ssid,
                });
              }
              if (systemInfo.board.rssi) {
                const signalStrength = systemInfo.board.rssi;
                let signalLevel = "弱";
                if (signalStrength > -50) signalLevel = "强";
                else if (signalStrength > -70) signalLevel = "中";

                tableData.push({
                  category: "网络信息",
                  property: "信号强度",
                  value: `${signalStrength} dBm (${signalLevel})`,
                });
              }
              if (systemInfo.board.channel) {
                tableData.push({
                  category: "网络信息",
                  property: "WiFi信道",
                  value: systemInfo.board.channel + "频道",
                });
              }
            }

            // 显示信息
            if (systemInfo.display) {
              if (systemInfo.display.width && systemInfo.display.height) {
                tableData.push({
                  category: "显示信息",
                  property: "屏幕尺寸",
                  value: `${systemInfo.display.width} × ${systemInfo.display.height}`,
                });
              }
              if (systemInfo.display.monochrome !== undefined) {
                tableData.push({
                  category: "显示信息",
                  property: "屏幕类型",
                  value: systemInfo.display.monochrome ? "单色屏" : "彩色屏",
                });
              }
            }

            // 其他信息
            if (systemInfo.uuid) {
              tableData.push({
                category: "设备信息",
                property: "设备UUID",
                value: systemInfo.uuid,
              });
            }

            if (systemInfo.language) {
              tableData.push({
                category: "设备信息",
                property: "系统语言",
                value: systemInfo.language,
              });
            }

            if (systemInfo.ota && systemInfo.ota.label) {
              tableData.push({
                category: "系统信息",
                property: "当前OTA分区",
                value: systemInfo.ota.label,
              });
            }
          } catch (parseError) {
            // 解析失败时显示原始信息
            tableData.push({
              category: "系统信息",
              property: "获取结果",
              value: result.success ? "获取成功，但解析失败" : "获取失败",
            });
          }
        } else {
          tableData.push({
            category: "系统信息",
            property: "获取结果",
            value: result.success ? "获取成功，但数据格式异常" : "获取失败",
          });
        }
      }

      return tableData;
    },
    async initTools() {
      if (!this.deviceId) {
        return;
      }

      this.toolsLoading = true;

      try {
        // 调用设备指令API获取工具列表
        const mcpRequest = {
          type: "mcp",
          payload: {
            jsonrpc: "2.0",
            id: 2,
            method: "tools/list",
            params: {
              withUserTools: true,
            },
          },
        };

        Api.device.sendDeviceCommand(this.deviceId, mcpRequest, ({ data }) => {
          this.toolsLoading = false;
          if (data.code === 0) {
            try {
              // 解析返回的工具列表数据
              const responseData = JSON.parse(data.data);

              // 检查两种可能的数据格式
              let tools = null;
              if (
                responseData &&
                responseData.payload &&
                responseData.payload.result &&
                responseData.payload.result.tools
              ) {
                // 标准MCP格式
                tools = responseData.payload.result.tools;
              } else if (
                responseData &&
                responseData.success &&
                responseData.data &&
                responseData.data.tools
              ) {
                // 设备返回的格式
                tools = responseData.data.tools;
              }

              if (tools && Array.isArray(tools) && tools.length > 0) {
                this.toolList = tools;
                // 默认选择第一个工具
                if (this.toolList.length > 0) {
                  this.selectedToolName = this.toolList[0].name;
                }
              } else {
                // 无法获取工具列表，显示空状态
                this.toolList = [];
              }
            } catch (error) {
              // 解析失败，显示空状态
              this.toolList = [];
            }
          } else {
            // API调用失败，显示空状态
            this.toolList = [];
          }
        });
      } catch (error) {
        this.toolsLoading = false;
        // 请求失败，显示空状态
        this.toolList = [];
      }
    },

    initToolParams() {
      // 清空现有参数
      this.toolParams = {};

      if (
        this.selectedTool &&
        this.selectedTool.inputSchema &&
        this.selectedTool.inputSchema.properties
      ) {
        // 使用 $nextTick 确保在下一个 tick 中设置参数值，避免响应式更新冲突
        this.$nextTick(() => {
          const newParams = {};
          Object.keys(this.selectedTool.inputSchema.properties).forEach((key) => {
            // 根据工具名称和参数名设置默认值
            if (
              this.selectedTool.name === "self.audio_speaker.set_volume" &&
              key === "volume"
            ) {
              newParams[key] = 100; // 音量默认值设为100
            } else if (
              this.selectedTool.name === "self.screen.set_brightness" &&
              key === "brightness"
            ) {
              newParams[key] = 100; // 亮度默认值设为100
            } else if (
              this.selectedTool.name === "self.screen.set_theme" &&
              key === "theme"
            ) {
              newParams[key] = "light"; // 主题默认值设为light
            } else {
              // 对于字符串类型的参数，设置为空字符串，对于数字类型设置为 null
              const property = this.selectedTool.inputSchema.properties[key];
              if (property.type === "string") {
                newParams[key] = "";
              } else if (property.type === "integer") {
                newParams[key] = property.minimum || 0;
              } else {
                newParams[key] = "";
              }
            }
          });

          // 一次性设置所有参数，避免多次触发响应式更新
          this.toolParams = { ...newParams };
        });
      }
      this.executionResult = null;
    },

    generateToolParamsRules() {
      this.toolParamsRules = {};
      if (
        this.selectedTool &&
        this.selectedTool.inputSchema &&
        this.selectedTool.inputSchema.properties
      ) {
        const requiredFields = this.selectedTool.inputSchema.required || [];

        Object.keys(this.selectedTool.inputSchema.properties).forEach((key) => {
          const property = this.selectedTool.inputSchema.properties[key];
          const rules = [];

          if (requiredFields.includes(key)) {
            rules.push({
              required: true,
              message: this.$t("mcpToolCall.requiredField", {
                field: this.formatPropertyLabel(key, property),
              }),
              trigger: "blur",
            });
          }

          if (property.type === "integer") {
            if (property.minimum !== undefined) {
              rules.push({
                validator: (rule, value, callback) => {
                  if (value < property.minimum) {
                    callback(
                      new Error(
                        this.$t("mcpToolCall.minValue", { value: property.minimum })
                      )
                    );
                  } else {
                    callback();
                  }
                },
                trigger: "blur",
              });
            }
            if (property.maximum !== undefined) {
              rules.push({
                validator: (rule, value, callback) => {
                  if (value > property.maximum) {
                    callback(
                      new Error(
                        this.$t("mcpToolCall.maxValue", { value: property.maximum })
                      )
                    );
                  } else {
                    callback();
                  }
                },
                trigger: "blur",
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
        volume: "音量",
        brightness: "亮度",
        theme: "主题",
        question: "问题",
        url: "网址",
        quality: "质量",
      };
      return labelMap[key] || key;
    },

    // 获取工具的显示名称
    getToolDisplayName(toolName) {
      const nameMap = {
        "self.get_device_status": "查看设备状态",
        "self.audio_speaker.set_volume": "设置音量",
        "self.screen.set_brightness": "设置亮度",
        "self.screen.set_theme": "设置主题",
        "self.camera.take_photo": "拍照识别",
        "self.get_system_info": "系统信息",
        "self.reboot": "重启设备",
        "self.upgrade_firmware": "升级固件",
        "self.screen.get_info": "屏幕信息",
        "self.screen.snapshot": "屏幕截图",
        "self.screen.preview_image": "预览图片",
        "self.assets.set_download_url": "设置下载地址",
      };
      return nameMap[toolName] || toolName;
    },

    // 获取工具分类
    getToolCategory(toolName) {
      if (toolName.includes("audio_speaker")) return "音频";
      if (toolName.includes("screen")) return "显示";
      if (toolName.includes("camera")) return "拍摄";
      if (
        toolName.includes("system") ||
        toolName.includes("reboot") ||
        toolName.includes("upgrade")
      )
        return "系统";
      if (toolName.includes("assets")) return "资源";
      return "设备信息";
    },

    // 获取简化的工具描述
    getSimpleDescription(originalDesc) {
      // 移除代码格式和复杂说明，保留核心功能描述
      return originalDesc.split("\n")[0].replace(/`/g, "");
    },

    // 获取工具帮助文本
    getToolHelpText(toolName) {
      const helpMap = {
        "self.get_device_status": "查看设备的当前运行状态，包括音量、屏幕、电池等信息。",
        "self.audio_speaker.set_volume": "调整设备的音量大小，请输入0-100之间的数值。",
        "self.screen.set_brightness": "调整设备屏幕的亮度，请输入0-100之间的数值。",
        "self.screen.set_theme": "切换设备屏幕的显示主题，可以选择浅色或深色模式。",
        "self.camera.take_photo":
          "使用设备摄像头拍摄照片并进行识别分析，请输入要询问的问题。",
        "self.get_system_info": "获取设备的系统信息，包括硬件规格、软件版本等。",
        "self.reboot": "重启设备，执行后设备将重新启动。",
        "self.upgrade_firmware": "从指定URL下载并升级设备固件，升级后设备会自动重启。",
        "self.screen.get_info": "获取屏幕的详细信息，如分辨率、尺寸等参数。",
        "self.screen.snapshot": "对当前屏幕进行截图并上传到指定URL。",
        "self.screen.preview_image": "在设备屏幕上预览指定URL的图片。",
        "self.assets.set_download_url": "设置设备资源文件的下载地址。",
      };
      return helpMap[toolName] || "";
    },

    executeTool() {
      if (!this.selectedTool) {
        this.$message.warning(this.$t("mcpToolCall.selectTool"));
        return;
      }

      // 验证必填参数
      const requiredFields = this.selectedTool.inputSchema.required || [];
      for (const field of requiredFields) {
        if (
          this.toolParams[field] === undefined ||
          this.toolParams[field] === null ||
          this.toolParams[field] === ""
        ) {
          this.$message.warning(
            this.$t("mcpToolCall.requiredField", {
              field: this.formatPropertyLabel(
                field,
                this.selectedTool.inputSchema.properties[field]
              ),
            })
          );
          return;
        }
      }

      // 构建MCP执行字符串
      const mcpExecuteString = {
        type: "mcp",
        payload: {
          jsonrpc: "2.0",
          id: 1,
          method: "tools/call",
          params: {
            name: this.selectedToolName,
            arguments: this.toolParams,
          },
        },
      };

      // 显示执行中状态
      this.executionResult = {
        request: mcpExecuteString,
      };

      // 调用设备执行工具
      Api.device.sendDeviceCommand(this.deviceId, mcpExecuteString, ({ data }) => {
        if (data.code === 0) {
          try {
            // 解析设备返回的结果
            const deviceResult = JSON.parse(data.data);

            // 检查是否为预定义工具，决定展示方式
            if (this.isPredefinedTool(this.selectedToolName)) {
              this.showResultAsTable = true;

              // 解析表格数据
              if (
                this.selectedToolName === "self.get_device_status" &&
                deviceResult.success &&
                deviceResult.data &&
                deviceResult.data.content &&
                deviceResult.data.content[0] &&
                deviceResult.data.content[0].text
              ) {
                try {
                  const deviceData = JSON.parse(deviceResult.data.content[0].text);
                  this.tableData = this.parseDeviceStatusToTable(deviceData);
                } catch (parseError) {
                  // 如果解析失败，回退到JSON模式
                  this.showResultAsTable = false;
                  this.tableData = [];
                }
              } else {
                // 其他预定义工具的表格解析
                this.tableData = this.parseOtherResultToTable(
                  this.selectedToolName,
                  deviceResult
                );
              }
            } else {
              // 非预定义工具，使用JSON模式
              this.showResultAsTable = false;
              this.tableData = [];
            }

            this.executionResult = {
              status: "success",
              response: deviceResult,
              timestamp: new Date().toLocaleString(),
            };
          } catch (error) {
            this.showResultAsTable = false;
            this.tableData = [];
            this.executionResult = {
              status: "error",
              request: mcpExecuteString,
              error: "解析设备响应失败: " + error.message,
              rawResponse: data.data,
              timestamp: new Date().toLocaleString(),
            };
          }
        } else {
          this.executionResult = {
            status: "error",
            request: mcpExecuteString,
            error: data.msg || "执行失败",
            timestamp: new Date().toLocaleString(),
          };
        }
      });
    },

    cancel() {
      this.closeDialog();
    },

    handleClose() {
      this.closeDialog();
    },

    closeDialog() {
      this.$emit("update:visible", false);
      this.selectedToolName = "";
      this.toolParams = {};
      this.toolParamsRules = {};
      this.toolSearchKeyword = "";
      this.executionResult = null;
      this.toolsLoading = false;
      this.showResultAsTable = false;
      this.tableData = [];
    },
  },
};
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

::v-deep .el-radio__input.is-checked + .el-radio__label {
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
  flex: 0.8;
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
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 200px;
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
  overflow: hidden;
}

.result-table {
  flex: 1;
  overflow-y: auto;
  min-height: 0;
}

.result-table ::v-deep .el-table {
  font-size: 12px;
}

.result-table ::v-deep .el-table__body-wrapper {
  max-height: none !important;
}

.result-table ::v-deep .el-table th {
  background: #f8f9fa;
  color: #606266;
  font-weight: 600;
  padding: 8px 0;
}

.result-table ::v-deep .el-table td {
  padding: 6px 0;
}

.result-text {
  flex: 1;
  margin: 0;
  padding: 8px;
  background: #f8f9fa;
  border: 1px solid #e4e7ed;
  border-radius: 4px;
  font-size: 12px;
  font-family: "Courier New", monospace;
  white-space: pre-wrap;
  word-wrap: break-word;
  overflow-y: auto;
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

/* 工具列表加载状态 */
.tool-list-loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 200px;
  color: #909399;
  text-align: center;
}

.tool-list-loading .el-icon-loading {
  font-size: 32px;
  color: #5778ff;
  margin-bottom: 12px;
  animation: rotating 1s linear infinite;
}

.loading-text {
  font-size: 14px;
  color: #606266;
}

@keyframes rotating {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
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
