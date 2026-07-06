# Session 管理：多会话隔离，每个 session 拥有独立的对话历史和上下文

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from minimal_agent.config import Config
from minimal_agent.context import ExecutionContext
from minimal_agent.llm import LLMProvider
from minimal_agent.loop import AgentLoop
from minimal_agent.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


@dataclass
class Session:
    id: str
    title: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    status: str = "active"  # "active" | "closed"


# Session 管理器：创建、切换、管理多个独立 session
class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    # 创建新 session
    def create(self, title: str = "") -> Session:
        sid = f"sess-{uuid.uuid4().hex[:8]}"
        session = Session(id=sid, title=title or f"会话 {len(self._sessions) + 1}")
        self._sessions[sid] = session
        self._locks[sid] = asyncio.Lock()
        log.info("Session 创建: %s", sid)
        return session

    # 获取 session，不存在返回 None
    def get(self, sid: str) -> Session | None:
        return self._sessions.get(sid)

    # 列出所有 session
    def list_all(self) -> list[Session]:
        return list(self._sessions.values())

    # 向 session 发送用户消息并执行 Agent 循环
    async def send_message(
        self,
        sid: str,
        content: str,
        provider: LLMProvider,
        registry: ToolRegistry,
        config: Config,
        *,
        on_event: Any = None,
    ) -> ExecutionContext:
        session = self._sessions.get(sid)
        if session is None:
            raise ValueError(f"Session not found: {sid}")

        lock = self._locks[sid]
        async with lock:
            # 追加用户消息到 session 历史
            session.messages.append({"role": "user", "content": content})

            # 构建 ExecutionContext（预填充 session 历史 + 本次输入）
            context = ExecutionContext(
                goal=content,
                max_steps=config.max_steps,
                max_context_tokens=config.max_context_tokens,
                compact_threshold=config.compact_threshold,
                messages=list(session.messages),
            )

            loop = AgentLoop(provider, registry, config, on_event=on_event)
            result = await loop.run(context)

            # 将本轮对话追加回 session 历史
            # 跳过第一条（重复的 user goal），只追加增量
            existing_count = len(session.messages)
            new_messages = result.messages[existing_count:]
            session.messages.extend(new_messages)

            return result

    # 关闭 session
    def close(self, sid: str) -> None:
        if sid in self._sessions:
            self._sessions[sid].status = "closed"
            log.info("Session 关闭: %s", sid)
