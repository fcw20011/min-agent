# Todo 工具：待办事项管理（内存存储）

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from minimal_agent.tools.base import BaseTool, ToolResult


@dataclass
class TodoItem:
    id: int
    content: str
    done: bool = False


# 全局内存存储（多 session 共享，生产环境应替换为持久化）
_todos: list[TodoItem] = []
_next_id: int = 1


class TodoAddTool(BaseTool):
    name = "todo_add"
    description = "添加一条新的待办事项。"
    input_schema = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "待办事项的内容",
            }
        },
        "required": ["content"],
    }

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        global _next_id
        content = str(params.get("content", "")).strip()
        if not content:
            return ToolResult(content="content is required", is_error=True)
        item = TodoItem(id=_next_id, content=content)
        _next_id += 1
        _todos.append(item)
        return ToolResult(content=f"已添加待办 #{item.id}: {content}")


class TodoListTool(BaseTool):
    name = "todo_list"
    description = "列出所有待办事项。"
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        if not _todos:
            return ToolResult(content="暂无待办事项。")
        lines = []
        for t in _todos:
            status = "✓" if t.done else "○"
            lines.append(f"  [{status}] #{t.id}: {t.content}")
        return ToolResult(content="待办列表:\n" + "\n".join(lines))


class TodoDoneTool(BaseTool):
    name = "todo_done"
    description = "将指定待办事项标记为已完成。"
    input_schema = {
        "type": "object",
        "properties": {
            "id": {
                "type": "integer",
                "description": "待办事项的编号",
            }
        },
        "required": ["id"],
    }

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        item_id = int(params.get("id", 0))
        for t in _todos:
            if t.id == item_id:
                t.done = True
                return ToolResult(content=f"已完成待办 #{item_id}: {t.content}")
        return ToolResult(content=f"未找到待办 #{item_id}", is_error=True)
