"""服务端插件工具执行器"""

from typing import Dict, Any, Optional, Tuple
from ..base import ToolType, ToolDefinition, ToolExecutor
from plugins_func.register import all_function_registry, Action, ActionResponse


class ServerPluginExecutor(ToolExecutor):
    """服务端插件工具执行器"""

    def __init__(self, conn):
        self.conn = conn
        self.config = conn.config
        # 存储知识库工具名称到真实插件名称的映射
        self._knowledge_base_mapping: Dict[str, str] = {}

    def _parse_knowledge_base_config(self, config_name: str) -> Optional[str]:
        """
        解析知识库配置名称，提取真实的插件名称

        Args:
            config_name: 配置名称，格式为 xzKnowledgeBase_<plugin_name>_<index>

        Returns:
            真实的插件名称，如果不是知识库配置则返回 None

        Example:
            "xzKnowledgeBase_search_from_ragflow_0" -> "search_from_ragflow"
            "xzKnowledgeBase_search_from_ragflow_1" -> "search_from_ragflow"
        """
        if not config_name.startswith("xzKnowledgeBase_"):
            return None

        # 移除前缀
        name_without_prefix = config_name[len("xzKnowledgeBase_"):]

        # 找到最后一个下划线的位置
        last_underscore_index = name_without_prefix.rfind("_")

        if last_underscore_index == -1:
            return None

        # 提取真实插件名称（从开头到最后一个下划线之前）
        real_plugin_name = name_without_prefix[:last_underscore_index]

        return real_plugin_name

    async def execute(
        self, conn, tool_name: str, arguments: Dict[str, Any]
    ) -> ActionResponse:
        """执行服务端插件工具"""
        # 检查是否是知识库工具调用
        real_plugin_name = self._knowledge_base_mapping.get(tool_name)
        if real_plugin_name:
            # 使用真实的插件名称获取函数
            func_item = all_function_registry.get(real_plugin_name)
        else:
            # 普通插件调用
            func_item = all_function_registry.get(tool_name)

        if not func_item:
            return ActionResponse(
                action=Action.NOTFOUND, response=f"插件函数 {tool_name} 不存在"
            )

        try:
            # 根据工具类型决定如何调用
            if hasattr(func_item, "type"):
                func_type = func_item.type
                if func_type.code in [4, 5]:  # SYSTEM_CTL, IOT_CTL (需要conn参数)
                    result = func_item.func(conn, **arguments)
                elif func_type.code == 2:  # WAIT
                    result = func_item.func(**arguments)
                elif func_type.code == 3:  # CHANGE_SYS_PROMPT
                    result = func_item.func(conn, **arguments)
                else:
                    result = func_item.func(**arguments)
            else:
                # 默认不传conn参数
                result = func_item.func(**arguments)

            return result

        except Exception as e:
            return ActionResponse(
                action=Action.ERROR,
                response=str(e),
            )

    def get_tools(self) -> Dict[str, ToolDefinition]:
        """获取所有注册的服务端插件工具"""
        tools = {}

        # 获取必要的函数
        necessary_functions = ["handle_exit_intent", "get_lunar"]

        # 获取配置中的函数
        config_functions = self.config["Intent"][
            self.config["selected_module"]["Intent"]
        ].get("functions", [])

        # 转换为列表
        if not isinstance(config_functions, list):
            try:
                config_functions = list(config_functions)
            except TypeError:
                config_functions = []

        # 合并所有需要的函数
        all_required_functions = list(set(necessary_functions + config_functions))

        for func_name in all_required_functions:
            func_item = all_function_registry.get(func_name)
            if func_item:
                tools[func_name] = ToolDefinition(
                    name=func_name,
                    description=func_item.description,
                    tool_type=ToolType.SERVER_PLUGIN,
                )

        # 处理知识库配置
        plugins_config = self.config.get("plugins", {})
        for config_name, config_value in plugins_config.items():
            # 检查是否是知识库配置
            real_plugin_name = self._parse_knowledge_base_config(config_name)
            if real_plugin_name:
                # 获取真实的插件函数
                func_item = all_function_registry.get(real_plugin_name)
                if func_item and isinstance(config_value, dict):
                    # 从配置中获取自定义的 name 和 description
                    custom_name = config_value.get("name", "")
                    custom_description = config_value.get("description", "")

                    # 创建动态的工具名称（使用配置名称去掉前缀部分作为工具名）
                    tool_name = config_name[len("xzKnowledgeBase_"):]

                    # 复制原始函数描述并修改
                    custom_func_desc = func_item.description.copy()
                    if "function" in custom_func_desc:
                        custom_func_desc["function"] = custom_func_desc["function"].copy()
                        custom_func_desc["function"]["name"] = tool_name
                        custom_func_desc["function"]["description"] = custom_description

                    # 注册工具
                    tools[tool_name] = ToolDefinition(
                        name=tool_name,
                        description=custom_func_desc,
                        tool_type=ToolType.SERVER_PLUGIN,
                    )

                    # 保存映射关系
                    self._knowledge_base_mapping[tool_name] = real_plugin_name

        return tools

    def has_tool(self, tool_name: str) -> bool:
        """检查是否有指定的服务端插件工具"""
        # 检查是否是知识库工具
        if tool_name in self._knowledge_base_mapping:
            return True
        # 检查是否是普通工具
        return tool_name in all_function_registry
