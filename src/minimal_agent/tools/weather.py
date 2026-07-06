# Weather 工具：模拟天气查询

from __future__ import annotations

import random
from typing import Any

from minimal_agent.tools.base import BaseTool, ToolResult


class WeatherTool(BaseTool):
    name = "weather"
    description = "查询指定城市的实时天气信息（模拟数据）。"
    input_schema = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称，如 '北京'",
            }
        },
        "required": ["city"],
    }

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        city = str(params.get("city", "")).strip()
        if not city:
            return ToolResult(content="city is required", is_error=True)

        # 模拟天气数据
        temp = random.randint(-5, 38)
        humidity = random.randint(20, 90)
        conditions = random.choice(["晴", "多云", "阴", "小雨", "中雨", "阵雨", "薄雾"])
        wind = random.choice(["无风", "微风", "和风", "强风"])

        return ToolResult(
            content=(
                f"[天气] {city}:\n"
                f"  温度: {temp}°C\n"
                f"  湿度: {humidity}%\n"
                f"  天气: {conditions}\n"
                f"  风力: {wind}"
            )
        )
