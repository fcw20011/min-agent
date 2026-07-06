# ReadFile 工具：读取本地文件内容

from __future__ import annotations

from pathlib import Path
from typing import Any

from minimal_agent.tools.base import BaseTool, ToolResult

MAX_BYTES = 50_000  # 最大读取 50KB，防止超大文件撑爆上下文


class ReadFileTool(BaseTool):
    name = "read_file"
    description = (
        "读取本地文件内容。支持文本文件和代码文件。"
        "自动限制最大读取大小以避免上下文溢出。"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件的绝对或相对路径",
            },
            "offset": {
                "type": "integer",
                "description": "从第几行开始读取（1-based），默认从头开始",
            },
            "limit": {
                "type": "integer",
                "description": "最多读取多少行，默认全部（有上限）",
            },
        },
        "required": ["path"],
    }

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        path_str = str(params.get("path", "")).strip()
        if not path_str:
            return ToolResult(content="path is required", is_error=True)

        file_path = Path(path_str).expanduser()
        if not file_path.exists():
            return ToolResult(content=f"文件不存在: {file_path}", is_error=True)
        if not file_path.is_file():
            return ToolResult(content=f"路径不是文件: {file_path}", is_error=True)

        # 检查文件大小
        size = file_path.stat().st_size
        if size > MAX_BYTES * 2:
            return ToolResult(
                content=f"文件过大 ({size} bytes)，超过读取上限。请使用 offset/limit 分段读取。",
                is_error=True,
            )

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return ToolResult(content=f"读取失败: {e}", is_error=True)

        # 截断
        if len(content) > MAX_BYTES:
            content = content[:MAX_BYTES] + "\n... [文件被截断，仅显示前 50KB]"

        lines = content.split("\n")
        offset = int(params.get("offset", 1))
        limit = int(params.get("limit", 0)) if params.get("limit") else 0

        if offset > 1:
            if offset > len(lines):
                return ToolResult(content=f"行号 {offset} 超出文件总行数 {len(lines)}", is_error=True)
            lines = lines[offset - 1:]

        if limit > 0:
            lines = lines[:limit]

        # 生成带行号的输出
        start_num = max(offset, 1)
        numbered = [f"{start_num + i:>4} | {line}" for i, line in enumerate(lines)]
        result_text = f"[{file_path}] ({len(lines)} 行):\n" + "\n".join(numbered)

        return ToolResult(content=result_text)
