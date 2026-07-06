# 工具基类：定义工具的名称、描述、参数 Schema 和执行接口

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = {}

    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> ToolResult:
        ...

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }
