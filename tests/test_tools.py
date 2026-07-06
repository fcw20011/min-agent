# 功能：验证所有工具的正确性和异常处理
# 设计：pytest 参数化测试覆盖正常输入、边界值和错误情况

import pytest
from minimal_agent.tools.calculator import CalculatorTool
from minimal_agent.tools.search import SearchTool
from minimal_agent.tools.weather import WeatherTool
from minimal_agent.tools.todo import TodoAddTool, TodoListTool, TodoDoneTool
from minimal_agent.tools.registry import ToolRegistry
from minimal_agent.tools.base import BaseTool, ToolResult


# ── Calculator ──
@pytest.mark.asyncio
async def test_calculator_simple():
    tool = CalculatorTool()
    result = await tool.execute({"expression": "3 + 5 * 2"})
    assert not result.is_error
    assert "13" in result.content or "13.0" in result.content


@pytest.mark.asyncio
async def test_calculator_power():
    tool = CalculatorTool()
    result = await tool.execute({"expression": "2 ^ 8"})
    assert not result.is_error
    assert "256" in result.content


@pytest.mark.asyncio
async def test_calculator_empty():
    tool = CalculatorTool()
    result = await tool.execute({"expression": ""})
    assert result.is_error


@pytest.mark.asyncio
async def test_calculator_invalid():
    tool = CalculatorTool()
    result = await tool.execute({"expression": "__import__('os')"})
    assert result.is_error


# ── Search ──
@pytest.mark.asyncio
async def test_search_python():
    tool = SearchTool()
    result = await tool.execute({"query": "Python"})
    assert not result.is_error
    assert "Python" in result.content


@pytest.mark.asyncio
async def test_search_unknown():
    tool = SearchTool()
    result = await tool.execute({"query": "xyz123nonexistent"})
    assert not result.is_error
    assert "搜索" in result.content


# ── Weather ──
@pytest.mark.asyncio
async def test_weather():
    tool = WeatherTool()
    result = await tool.execute({"city": "北京"})
    assert not result.is_error
    assert "北京" in result.content
    assert "°C" in result.content


@pytest.mark.asyncio
async def test_weather_empty():
    tool = WeatherTool()
    result = await tool.execute({"city": ""})
    assert result.is_error


# ── Todo ──
@pytest.mark.asyncio
async def test_todo_add_list_done():
    add = TodoAddTool()
    lst = TodoListTool()
    done = TodoDoneTool()

    r1 = await add.execute({"content": "写周报"})
    assert not r1.is_error

    r2 = await lst.execute({})
    assert not r2.is_error
    assert "写周报" in r2.content

    r3 = await done.execute({"id": 1})
    assert not r3.is_error
    assert "已完成" in r3.content


# ── Registry ──
def test_registry_register_and_get():
    registry = ToolRegistry()
    tool = CalculatorTool()
    registry.register(tool)
    assert registry.get("calculator") is tool
    assert registry.get("nonexistent") is None


def test_registry_schemas():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(SearchTool())
    schemas = registry.schemas()
    assert len(schemas) == 2
    names = {s["name"] for s in schemas}
    assert names == {"calculator", "search"}
    for s in schemas:
        assert "description" in s
        assert "parameters" in s


# ── BaseTool Schema ──
class _FakeTool(BaseTool):
    name = "fake"
    description = "fake tool"
    input_schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}

    async def execute(self, params):
        return ToolResult(content="ok")


def test_tool_schema_format():
    tool = _FakeTool()
    s = tool.schema()
    assert s["name"] == "fake"
    assert s["description"] == "fake tool"
    assert s["parameters"]["properties"]["x"]["type"] == "string"
