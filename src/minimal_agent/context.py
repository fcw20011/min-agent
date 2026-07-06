# ExecutionContext：管理单次 Agent Run 的所有状态，包括消息历史、步数、状态跟踪

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    goal: str
    max_steps: int
    max_context_tokens: int = 8000
    compact_threshold: float = 0.75

    # 运行时状态
    messages: list[dict[str, Any]] = field(default_factory=list)
    step: int = 0
    status: str = "running"  # "running" | "success" | "failed"
    reason: str | None = None
    result: str = ""

    # 可选的系统提示词覆盖（用于 Skill）
    system_prompt_override: str | None = None

    def __post_init__(self) -> None:
        # 初始化第一条用户消息
        if not self.messages:
            self.messages = [{"role": "user", "content": self.goal}]

    # 返回 system prompt，优先使用 override
    def system_prompt(self, base: str) -> str:
        return self.system_prompt_override or base

    # 将 LLM 响应的 content blocks 追加为 assistant 消息
    def add_assistant_message(self, blocks: list[dict[str, Any]]) -> None:
        self.messages.append({"role": "assistant", "content": blocks})

    # 将工具调用结果追加为 tool 消息（同一 step 的多个结果合并）
    def add_tool_result(self, tool_call_id: str, content: str, is_error: bool = False) -> None:
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })

    # 检查是否应当终止循环
    def is_done(self) -> bool:
        return self.status != "running"

    def mark_success(self) -> None:
        self.status = "success"

    def mark_failed(self, reason: str) -> None:
        self.status = "failed"
        self.reason = reason

    # 获取当前上下文估算 token 数（粗估：字符数 / 3.5）
    def estimate_tokens(self) -> int:
        total_chars = sum(len(str(m)) for m in self.messages)
        return int(total_chars / 3.5)

    # 检查是否需要压缩上下文
    def should_compact(self) -> bool:
        estimated = self.estimate_tokens()
        threshold_tokens = int(self.max_context_tokens * self.compact_threshold)
        return estimated > threshold_tokens


# 简单压缩：保留系统提示 + 首条用户消息 + 最近 N 轮，其余用摘要替换
async def compact_context(
    context: ExecutionContext,
    provider: Any,  # LLMProvider
    keep_last: int = 4,
) -> None:
    if len(context.messages) <= keep_last + 2:
        return

    # 收集要压缩的中间消息
    messages_to_compact = context.messages[1:-(keep_last)]
    if not messages_to_compact:
        return

    # 拼接被压缩的消息
    compact_text = ""
    for m in messages_to_compact:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        compact_text += block.get("text", "") + "\n"
                    elif block.get("type") == "tool_use":
                        compact_text += f"[调用工具 {block.get('name', '?')}]\n"
                else:
                    compact_text += str(block) + "\n"
        else:
            compact_text += f"[{role}] {str(content)[:200]}\n"

    # 调用 LLM 生成摘要
    try:
        summary_response = await provider.chat(
            messages=[{"role": "user", "content": f"请用中文简要总结以下对话的关键信息（不超过200字）：\n\n{compact_text}"}],
            tool_schemas=None,
            system="你是一个对话摘要助手，请简洁总结。",
        )
        summary = summary_response.text.strip() or "（对话上下文）"
    except Exception:
        log.warning("压缩摘要生成失败，使用截断方式")
        summary = f"[上下文压缩] 以下为 {len(messages_to_compact)} 条历史消息的摘要"

    # 替换上下文：保留系统消息 + 首条用户消息 + 摘要 + 最近 N 轮
    first_user = context.messages[0] if context.messages else None
    recent = context.messages[-keep_last:]

    new_messages: list[dict[str, Any]] = []
    if first_user is not None:
        new_messages.append(first_user)
    new_messages.append({"role": "user", "content": f"[对话摘要] {summary}"})
    new_messages.append({"role": "assistant", "content": "已理解之前的对话内容，我会继续。请问还有什么需要帮助的吗？"})
    new_messages.extend(recent)
    context.messages = new_messages
    log.info("上下文已压缩：%d 条消息 → %d 条", len(messages_to_compact) + keep_last + 1, len(new_messages))
