from plugins_func.register import register_function, ToolType, ActionResponse, Action

# 定义基础的函数描述模板
SEARCH_FROM_RAGFLOW_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "search_from_ragflow",
        "description": "从知识库中查询信息",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "查询的关键词"}},
            "required": ["query"],
        },
    },
}


@register_function(
    "search_from_ragflow", SEARCH_FROM_RAGFLOW_FUNCTION_DESC, ToolType.WAIT
)
def search_from_ragflow(query=None):
    """
    用于从ragflow知识库中查询信息
    """
    # TODO 从ragflow知识库中查询信息
    if query and "医生" in query:
        response_text = "医院有张山、里斯、王五3名全科医生，其中王五医生是主要擅长眼科"
    elif query and "科室" in query:
        response_text = "医院眼科、麻醉科"
    else:
        response_text = "暂无相关信息"

    return ActionResponse(Action.REQLLM, response_text, None)
