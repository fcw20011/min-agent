# 功能：验证 ExecutionContext 的状态管理和压缩逻辑
# 设计：构造特定消息序列，验证 token 估算、终止状态标记和压缩后的消息结构

import pytest
from minimal_agent.context import ExecutionContext, compact_context


def test_context_init():
    ctx = ExecutionContext(goal="测试目标", max_steps=10)
    assert ctx.status == "running"
    assert ctx.step == 0
    assert len(ctx.messages) == 1
    assert ctx.messages[0]["role"] == "user"
    assert "测试目标" in str(ctx.messages[0]["content"])


def test_context_add_messages():
    ctx = ExecutionContext(goal="test", max_steps=10)
    ctx.add_assistant_message([{"type": "text", "text": "Hello"}])
    assert len(ctx.messages) == 2
    assert ctx.messages[1]["role"] == "assistant"

    ctx.add_tool_result("tc_1", "result 1")
    assert len(ctx.messages) == 3
    assert ctx.messages[2]["role"] == "tool"
    assert ctx.messages[2]["content"] == "result 1"


def test_context_mark_states():
    ctx = ExecutionContext(goal="test", max_steps=10)
    assert not ctx.is_done()

    ctx.mark_success()
    assert ctx.is_done()
    assert ctx.status == "success"

    ctx2 = ExecutionContext(goal="test", max_steps=10)
    ctx2.mark_failed("timeout")
    assert ctx2.is_done()
    assert ctx2.status == "failed"
    assert ctx2.reason == "timeout"


def test_context_estimate_tokens():
    ctx = ExecutionContext(goal="hello world", max_steps=10)
    tokens = ctx.estimate_tokens()
    assert tokens > 0


def test_context_should_compact():
    ctx = ExecutionContext(goal="test", max_steps=10, max_context_tokens=100, compact_threshold=0.5)
    # 初始状态不应压缩
    assert not ctx.should_compact()

    # 添加大量消息触发压缩阈值
    ctx.messages.append({"role": "assistant", "content": "x" * 300})
    assert ctx.should_compact()


def test_context_system_prompt():
    ctx = ExecutionContext(goal="test", max_steps=10)
    assert "You are" in ctx.system_prompt("You are helpful.")

    ctx2 = ExecutionContext(goal="test", max_steps=10, system_prompt_override="自定义提示词")
    assert ctx2.system_prompt("default") == "自定义提示词"


# ── 压缩测试（不依赖真实 LLM）──
class _FakeProvider:
    async def chat(self, messages, tool_schemas=None, system=""):
        from minimal_agent.llm import LLMResponse
        return LLMResponse(text="这是一个摘要")


@pytest.mark.asyncio
async def test_compact_context_basic():
    ctx = ExecutionContext(goal="test", max_steps=10)
    ctx.messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "how are you"},
        {"role": "assistant", "content": "fine"},
        {"role": "user", "content": "what is python"},
        {"role": "assistant", "content": "a programming language"},
        {"role": "user", "content": "tell me more"},
        {"role": "assistant", "content": "sure"},
    ]
    original_len = len(ctx.messages)
    await compact_context(ctx, _FakeProvider(), keep_last=4)
    # 压缩后：首条用户消息 + 摘要 + 确认 + 最近 4 轮
    assert len(ctx.messages) < original_len
    assert "已理解之前的对话内容" in str(ctx.messages)
