# Search 工具：模拟网络搜索（mock）

from __future__ import annotations

from typing import Any

from minimal_agent.tools.base import BaseTool, ToolResult


# 模拟搜索结果数据
_MOCK_DATA = {
    "北京天气": "北京今日晴，25°C，湿度40%，风力2级，适合户外活动。",
    "上海天气": "上海今日多云转阴，22°C，湿度65%，可能有小雨。",
    "Python": "Python 是一种解释型、面向对象的高级编程语言，由 Guido van Rossum 于 1991 年发布。广泛应用于 Web 开发、数据科学、AI 等领域。",
    "Agent": "AI Agent 是一种能够自主感知环境、制定计划、调用工具并执行任务的智能体系统。核心架构包含感知、规划、执行、记忆模块。",
    "OpenAI": "OpenAI 是一家美国 AI 研究公司，成立于 2015 年。开发了 GPT 系列大语言模型、DALL-E 图像生成模型、Sora 视频生成模型等产品。",
    "default": "搜索到相关结果：关于此话题的详细信息需要进一步验证。建议参考官方文档或权威来源获取最新信息。",
}


class SearchTool(BaseTool):
    name = "search"
    description = "搜索互联网信息（当前为模拟实现，返回预置数据）。获取事实、定义、新闻等公开信息。"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词或问题",
            }
        },
        "required": ["query"],
    }

    # 从模拟数据中检索最匹配的结果
    async def execute(self, params: dict[str, Any]) -> ToolResult:
        query = str(params.get("query", "")).strip()
        if not query:
            return ToolResult(content="query is required", is_error=True)

        # 模糊匹配
        for key, value in _MOCK_DATA.items():
            if key in query or query in key:
                return ToolResult(content=f"[搜索] {query}:\n{value}")

        return ToolResult(content=f"[搜索] {query}:\n{_MOCK_DATA['default']}")
