# 功能：验证 AgentLoop 的完整执行流程（使用 mock LLM）
# 设计：构造 FakeLLMProvider 模拟 LLM 响应，验证循环终止条件和消息结构

import pytest
from minimal_agent.config import Config
from minimal_agent.context import ExecutionContext
from minimal_agent.llm import LLMResponse, ToolCallBlock
from minimal_agent.loop import AgentLoop
from minimal_agent.tools.registry import ToolRegistry
from minimal_agent.tools.calculator import CalculatorTool
from minimal_agent.tools.weather import WeatherTool


class _FakeLLM:
    """模拟 LLM：第一次返回工具调用，第二次返回最终回复"""

    def __init__(self):
        self._call_count = 0

    async def chat(self, messages, tool_schemas=None, system=""):
        self._call_count += 1
        if self._call_count == 1:
            # 第一次：调用 calculator
            return LLMResponse(
                text="",
                tool_calls=[ToolCallBlock(id="tc_1", name="calculator", input={"expression": "3 + 5"})],
                stop_reason="tool_use",
                usage={"input_tokens": 50, "output_tokens": 10},
            )
        else:
            # 第二次：最终回复
            return LLMResponse(
                text="计算结果是 8。",
                stop_reason="end_turn",
                usage={"input_tokens": 100, "output_tokens": 20},
            )


@pytest.mark.asyncio
async def test_loop_tool_call_then_answer():
    config = Config(max_steps=5)
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    provider = _FakeLLM()
    context = ExecutionContext(goal="计算 3+5", max_steps=5)

    loop = AgentLoop(provider, registry, config)
    result = await loop.run(context)

    assert result.status == "success"
    assert result.step == 2
    assert "计算结果是 8" in result.result


class _FakeLLM_DirectAnswer:
    async def chat(self, messages, tool_schemas=None, system=""):
        return LLMResponse(
            text="你好！有什么可以帮助你的吗？",
            stop_reason="end_turn",
            usage={"input_tokens": 20, "output_tokens": 10},
        )


@pytest.mark.asyncio
async def test_loop_direct_answer():
    config = Config(max_steps=5)
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    provider = _FakeLLM_DirectAnswer()
    context = ExecutionContext(goal="你好", max_steps=5)

    loop = AgentLoop(provider, registry, config)
    result = await loop.run(context)

    assert result.status == "success"
    assert result.step == 1


class _FakeLLM_MaxSteps:
    def __init__(self):
        self._count = 0

    async def chat(self, messages, tool_schemas=None, system=""):
        self._count += 1
        return LLMResponse(
            text="",
            tool_calls=[ToolCallBlock(id=f"tc_{self._count}", name="calculator", input={"expression": "1+1"})],
            stop_reason="tool_use",
            usage={"input_tokens": 20, "output_tokens": 10},
        )


@pytest.mark.asyncio
async def test_loop_exceed_max_steps():
    config = Config(max_steps=3)
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    provider = _FakeLLM_MaxSteps()
    context = ExecutionContext(goal="无限循环", max_steps=3)

    loop = AgentLoop(provider, registry, config)
    result = await loop.run(context)

    assert result.status == "failed"
    assert result.reason == "exceeded_max_steps"
    assert result.step == 3


class _FakeLLM_UnknownTool:
    async def chat(self, messages, tool_schemas=None, system=""):
        return LLMResponse(
            text="",
            tool_calls=[ToolCallBlock(id="tc_1", name="unknown_tool", input={})],
            stop_reason="tool_use",
            usage={"input_tokens": 20, "output_tokens": 10},
        )


@pytest.mark.asyncio
async def test_loop_unknown_tool():
    config = Config(max_steps=5)
    registry = ToolRegistry()  # 空注册表

    provider = _FakeLLM_UnknownTool()
    context = ExecutionContext(goal="用未知工具", max_steps=5)

    loop = AgentLoop(provider, registry, config)
    result = await loop.run(context)

    # 未知工具不应崩溃，应返回错误结果，然后可能继续或终止
    assert result.status in ("running", "success", "failed")
