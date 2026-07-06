# 工具注册表：维护所有已注册工具的字典，提供注册、查找和 Schema 导出

from __future__ import annotations

from typing import Any

from minimal_agent.tools.base import BaseTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    # 注册工具，同名覆盖
    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    # 按名称查找工具，不存在返回 None
    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    # 导出所有工具的 Function Calling Schema 列表
    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)
