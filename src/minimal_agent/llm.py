# LLM Provider：统一接口，底层适配 OpenAI 兼容 API 和 Anthropic 原生 API
from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from minimal_agent.config import Config

log = logging.getLogger(__name__)


@dataclass
class ToolCallBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    thinking: str = ""
    # Anthropic thinking 块的 signature，下次请求必须原样回传
    thinking_signature: str = ""
    tool_calls: list[ToolCallBlock] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: dict[str, Any] = field(default_factory=dict)


class LLMProvider:
    # 初始化客户端，根据 provider 类型选择 SDK
    def __init__(self, config: Config):
        self._config = config

        if config.is_anthropic:
            import anthropic
            self._client: Any = anthropic.AsyncAnthropic(api_key=config.api_key, base_url=config.base_url)
        else:
            from openai import AsyncOpenAI
            extra_kwargs: dict[str, Any] = {}
            if config.extra_headers:
                extra_kwargs["default_headers"] = config.extra_headers
            self._client = AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                **extra_kwargs,
            )

    # ── 统一入口 → 按 provider 分发 ──
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, object]] | None = None,
        system: str = "You are a helpful AI assistant.",
    ) -> LLMResponse:
        if self._config.is_anthropic:
            return await self._chat_anthropic(messages, tool_schemas or [], system)
        else:
            return await self._chat_openai(messages, tool_schemas or [], system)

    # ══════════════════════════════════════════════
    #  OpenAI 兼容路径（openai / deepseek / xiaomi）
    # ══════════════════════════════════════════════
    async def _chat_openai(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, object]],
        system: str,
    ) -> LLMResponse:
        from openai import AsyncOpenAI
        client: AsyncOpenAI = self._client

        api_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        api_messages.extend(messages)

        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": api_messages,
            "temperature": 0.7,
        }
        tools = [{"type": "function", "function": ts} for ts in tool_schemas]
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        log.debug("OpenAI request: model=%s msgs=%d tools=%d",
                  self._config.model, len(api_messages), len(tools))

        response = await client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        msg = choice.message
        finish_reason = choice.finish_reason or "stop"

        result = LLMResponse(
            text=msg.content or "",
            stop_reason="tool_use" if finish_reason == "tool_calls" else "end_turn",
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        )

        # DeepSeek R1 reasoning_content
        if self._config.provider == "deepseek":
            reasoning = getattr(msg, "reasoning_content", None)
            if reasoning:
                result.thinking = str(reasoning)

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    input_data = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    input_data = {}
                result.tool_calls.append(ToolCallBlock(id=tc.id, name=tc.function.name, input=input_data))

        result.usage["context_pct"] = self._context_pct(messages)
        return result

    # ══════════════════════════════════════════════
    #  Anthropic 原生路径
    # ══════════════════════════════════════════════
    async def _chat_anthropic(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, object]],
        system: str,
    ) -> LLMResponse:
        import anthropic

        # ── 分离 system 并构造 Anthropic system blocks ──
        system_blocks: list[dict[str, object]] = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
        ]

        # ── 转换消息格式 ──
        anthropic_messages = self._to_anthropic_messages(messages)

        # ── 转换工具 schema：parameters → input_schema ──
        anthropic_tools: list[dict[str, object]] = []
        for ts in tool_schemas:
            anthropic_tools.append({
                "name": ts["name"],
                "description": ts.get("description", ""),
                "input_schema": ts.get("parameters", ts.get("input_schema", {})),
            })
        # 最后一个 tool 加 cache_control
        if anthropic_tools:
            anthropic_tools[-1] = dict(anthropic_tools[-1])
            anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}

        kwargs: dict[str, object] = {
            "model": self._config.model,
            "max_tokens": 8192,
            "system": system_blocks,
            "messages": anthropic_messages,
        }
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        log.debug("Anthropic request: model=%s msgs=%d tools=%d",
                  self._config.model, len(anthropic_messages), len(anthropic_tools))

        response = await self._client.messages.create(**kwargs)

        # ── 解析响应 ──
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        thinking_sig = ""

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                # Anthropic 的 input 已是 dict
                self._pending_tool_calls = getattr(self, "_pending_tool_calls", [])
                # 直接用实例变量收集
                pass  # 在下面统一处理
            elif block.type == "thinking":
                thinking_parts.append(block.thinking)
                thinking_sig = getattr(block, "signature", "")

        # 收集 tool_use
        tool_calls: list[ToolCallBlock] = []
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append(ToolCallBlock(id=block.id, name=block.name, input=dict(block.input)))

        stop_reason = response.stop_reason or "end_turn"
        # Anthropic stop_reason 映射
        if stop_reason == "tool_use":
            internal_reason = "tool_use"
        elif stop_reason in ("end_turn", "stop_sequence"):
            internal_reason = "end_turn"
        else:
            internal_reason = stop_reason

        usage = response.usage
        return LLMResponse(
            text="".join(text_parts),
            thinking="".join(thinking_parts),
            thinking_signature=thinking_sig,
            tool_calls=tool_calls,
            stop_reason=internal_reason,
            usage={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0),
                "cache_create_tokens": getattr(usage, "cache_creation_input_tokens", 0),
                "context_pct": usage.input_tokens / self._config.resolve_model_max_tokens(),
            },
        )

    # ── 内部消息 → Anthropic 格式 ──
    def _to_anthropic_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """将内部消息格式转为 Anthropic 兼容格式。

        内部格式:
          {"role": "user", "content": "hello"}
          {"role": "assistant", "content": [{"type": "text", ...}, {"type": "tool_use", ...}]}
          {"role": "tool", "tool_call_id": "...", "content": "result"}

        Anthropic 格式:
          {"role": "user", "content": "hello"}
          {"role": "assistant", "content": [{"type": "text", ...}, {"type": "tool_use", ...}]}
          {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "...", "content": "..."}]}
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "")
            # 跳过 system 消息（已作为顶层参数传递）
            if role == "system":
                continue

            if role == "tool":
                # 转为 user + tool_result block
                result.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": str(msg.get("content", "")),
                        }
                    ],
                })
            elif role == "assistant" and isinstance(msg.get("content"), list):
                # 保留 content blocks，但要确保 thinking 含 signature
                blocks = deepcopy(msg["content"])
                result.append({"role": "assistant", "content": blocks})
            else:
                # user 消息或其他，直接传
                result.append(dict(msg))

        return result

    # ── token 估算 ──
    def _context_pct(self, messages: list[dict[str, Any]]) -> float:
        total_chars = sum(len(str(m)) for m in messages)
        max_tokens = self._config.resolve_model_max_tokens()
        return min(total_chars / (max_tokens * 3.5), 1.0)
