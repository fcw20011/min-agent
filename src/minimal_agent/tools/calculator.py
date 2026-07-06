# Calculator 工具：安全的数学表达式求值

from __future__ import annotations

import operator
from typing import Any

from minimal_agent.tools.base import BaseTool, ToolResult


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "安全计算数学表达式。支持加减乘除、幂运算、括号等基本运算。"
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，如 '(3 + 5) * 2' 或 'sqrt(144)'",
            }
        },
        "required": ["expression"],
    }

    # 白名单方式安全求值，禁止任意代码执行
    async def execute(self, params: dict[str, Any]) -> ToolResult:
        expr = str(params.get("expression", ""))
        if not expr:
            return ToolResult(content="expression is required", is_error=True)

        # 安全：仅允许数字、运算符、括号、空格、小数点
        allowed = set("0123456789+-*/^().% eE")
        cleaned = "".join(c for c in expr if c in allowed)
        if cleaned != expr.strip():
            return ToolResult(
                content=f"表达式包含不安全的字符，已过滤为: {cleaned}。如需使用 sqrt/abs 等函数请用数学公式替代。",
                is_error=True,
            )

        expr = cleaned.replace("^", "**")
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return ToolResult(content=f"计算结果: {result}")
        except Exception as e:
            return ToolResult(content=f"计算错误: {e}", is_error=True)
