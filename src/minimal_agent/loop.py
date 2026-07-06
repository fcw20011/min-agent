# AgentLoop：plan → act → observe 核心循环

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any, Callable

from minimal_agent.config import Config
from minimal_agent.context import ExecutionContext, compact_context
from minimal_agent.llm import LLMProvider, LLMResponse, ToolCallBlock
from minimal_agent.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class AgentLoop:
    # 初始化循环所需依赖：LLM provider、工具注册表、配置
    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        config: Config,
        *,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        self._provider = provider
        self._registry = registry
        self._config = config
        self._on_event = on_event

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        data["type"] = event_type
        data["ts"] = _now()
        if self._on_event:
            self._on_event(event_type, data)

    # 驱动 plan→act→observe 循环直到上下文终止
    async def run(self, context: ExecutionContext) -> ExecutionContext:
        tool_schemas = self._registry.schemas()

        while not context.is_done():
            context.step += 1
            self._emit("step_started", {"step": context.step, "messages": len(context.messages)})

            # ── [plan] 调用 LLM ──
            try:
                response = await self._provider.chat(
                    messages=context.messages,
                    tool_schemas=tool_schemas,
                    system=context.system_prompt(
                        "你是一个有用的 AI 助手。可以使用工具来完成任务。"
                        "当目标已达成时，直接回复最终结果，不要继续调用工具。"
                        "如果需要计算，使用 calculator 工具。"
                        "如果需要获取信息，使用 search 或 weather 工具。"
                        "如果需要管理任务，使用 todo_add/todo_list/todo_done 工具。"
                    ),
                )
            except asyncio.CancelledError:
                context.mark_failed("cancelled")
                raise
            except Exception:
                log.exception("LLM 调用失败 step=%d", context.step)
                context.mark_failed("llm_error")
                break

            # ── [observe] 追加 assistant 响应 ──
            blocks: list[dict[str, Any]] = []
            if response.thinking:
                blocks.append({"type": "thinking", "thinking": response.thinking})
            if response.text:
                blocks.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                })
            context.add_assistant_message(blocks)

            # ── [act] 执行工具调用 ──
            if response.stop_reason == "tool_use":
                for tc in response.tool_calls:
                    result = await self._invoke_tool(tc)
                    context.add_tool_result(tc.id, result.content, is_error=result.is_error)

            # ── 终止检查 ──
            if response.stop_reason == "end_turn":
                context.result = response.text or ""
                context.mark_success()
            elif context.step >= context.max_steps:
                context.mark_failed("exceeded_max_steps")

            # ── 压缩检查 ──
            if (
                not context.is_done()
                and context.should_compact()
                and self._config.compact_threshold > 0
            ):
                await compact_context(context, self._provider)

            self._emit("step_finished", {
                "step": context.step,
                "stop_reason": response.stop_reason,
                "context_tokens": context.estimate_tokens(),
            })

        self._emit("run_finished", {
            "status": context.status,
            "steps": context.step,
            "reason": context.reason,
        })

        return context

    # 执行单个工具调用，带异常处理和超时保护
    async def _invoke_tool(self, tc: ToolCallBlock) -> Any:
        from minimal_agent.tools.base import ToolResult

        tool = self._registry.get(tc.name)
        if tool is None:
            msg = f"未知工具: {tc.name}"
            log.warning(msg)
            return ToolResult(content=msg, is_error=True)

        t0 = time.monotonic()
        self._emit("tool_call_started", {
            "tool_name": tc.name,
            "tool_use_id": tc.id,
            "params": tc.input,
        })

        try:
            result = await asyncio.wait_for(tool.execute(tc.input), timeout=60.0)
        except asyncio.TimeoutError:
            result = ToolResult(content=f"工具 {tc.name} 执行超时", is_error=True)
        except Exception as e:
            log.exception("工具 %s 执行异常", tc.name)
            result = ToolResult(content=f"工具执行错误: {e}", is_error=True)

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        self._emit("tool_call_finished", {
            "tool_name": tc.name,
            "tool_use_id": tc.id,
            "elapsed_ms": elapsed_ms,
            "is_error": result.is_error,
            "output_preview": result.content[:200],
        })

        return result
