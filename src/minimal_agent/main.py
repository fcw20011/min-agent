# CLI 主入口：交互式终端界面，支持多 session 切换、日志输出和 trace

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from minimal_agent.config import Config, load_config
from minimal_agent.llm import LLMProvider
from minimal_agent.session import SessionManager
from minimal_agent.tools.calculator import CalculatorTool
from minimal_agent.tools.read_file import ReadFileTool
from minimal_agent.tools.registry import ToolRegistry
from minimal_agent.tools.search import SearchTool
from minimal_agent.tools.todo import TodoAddTool, TodoDoneTool, TodoListTool
from minimal_agent.tools.weather import WeatherTool

_logger: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = logging.getLogger("minimal_agent")
        _logger.setLevel(logging.DEBUG)
        # 文件 handler：完整 trace
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        fh = logging.FileHandler(log_dir / "agent.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        _logger.addHandler(fh)
    return _logger


def _build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(ReadFileTool())
    registry.register(SearchTool())
    registry.register(WeatherTool())
    registry.register(TodoAddTool())
    registry.register(TodoListTool())
    registry.register(TodoDoneTool())
    return registry


def _event_handler(event_type: str, data: dict) -> None:
    log = _get_logger()
    if event_type == "step_started":
        print(f"\n  [Step {data['step']}] ", end="", flush=True)
    elif event_type == "tool_call_started":
        print(f"\r  → 调用工具: {data['tool_name']}({json.dumps(data['params'], ensure_ascii=False)})")
        log.info("工具调用开始: %s %s", data["tool_name"], data["params"])
    elif event_type == "tool_call_finished":
        status = "✗" if data.get("is_error") else "✓"
        print(f"  {status} 工具完成: {data['tool_name']} ({data['elapsed_ms']}ms)")
        log.info("工具调用完成: %s error=%s elapsed=%dms", data["tool_name"], data["is_error"], data["elapsed_ms"])
    elif event_type == "step_finished":
        tokens = data.get("context_tokens", 0)
        print(f"  ── 上下文: ~{tokens} tokens", end="", flush=True)
    elif event_type == "run_finished":
        print(f"\n  [{data['status']}] 完成，共 {data['steps']} 步")
        log.info("Run 完成: status=%s steps=%d reason=%s", data["status"], data["steps"], data.get("reason"))


def _print_help() -> None:
    print("""
  命令:
    /help         显示帮助
    /sessions     列出所有 session
    /new [标题]   创建新 session
    /switch <id>  切换到指定 session
    /close        关闭当前 session
    /trace <id>   显示指定 run 的完整日志路径
    /exit         退出

  工具:
    calculator    数学计算
    search        信息搜索（模拟）
    weather       天气查询（模拟）
    todo_add      添加待办
    todo_list     列出待办
    todo_done     完成待办
""")


async def _run_interactive(config: Config) -> None:
    log = _get_logger()
    provider = LLMProvider(config)
    registry = _build_registry()
    sessions = SessionManager()

    # 创建默认 session
    current = sessions.create("默认会话")
    print(f"  Session: {current.id} ({current.title})")
    print("  输入消息开始对话，或输入 /help 查看命令")

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见！")
            break

        if not user_input:
            continue

        # 处理命令
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/exit":
                break
            elif cmd == "/help":
                _print_help()
            elif cmd == "/sessions":
                for s in sessions.list_all():
                    marker = "←" if s.id == current.id else " "
                    print(f"  [{marker}] {s.id}: {s.title} ({s.status})")
            elif cmd == "/new":
                current = sessions.create(arg)
                print(f"  → 切换到: {current.id} ({current.title})")
            elif cmd == "/switch":
                target = sessions.get(arg)
                if target:
                    current = target
                    print(f"  → 切换到: {current.id} ({current.title})")
                else:
                    print(f"  Session 不存在: {arg}")
            elif cmd == "/close":
                sessions.close(current.id)
                remaining = [s for s in sessions.list_all() if s.status == "active"]
                if remaining:
                    current = remaining[0]
                    print(f"  → 切换到: {current.id} ({current.title})")
                else:
                    current = sessions.create("新会话")
                    print(f"  → 创建: {current.id}")
            elif cmd == "/trace":
                print(f"  Trace 日志: {Path('logs/agent.log').absolute()}")
            else:
                print(f"  未知命令: {cmd}，输入 /help 查看帮助")
            continue

        # 发送消息到当前 session
        try:
            context = await sessions.send_message(
                current.id,
                user_input,
                provider,
                registry,
                config,
                on_event=_event_handler,
            )

            # 打印最终响应
            if context.status == "success":
                final_msg = context.messages
                # 找到最后一条 assistant 的文本内容
                for m in reversed(final_msg):
                    if m["role"] == "assistant":
                        content = m.get("content", "")
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    print(f"\n  Agent: {block['text']}")
                                    break
                        elif isinstance(content, str) and content:
                            print(f"\n  Agent: {content}")
                        break
            else:
                print(f"\n  [错误] Run 失败: {context.reason}")

        except Exception as e:
            log.exception("Run 异常")
            print(f"\n  [错误] {e}")


def main() -> None:
    config = load_config()

    if not config.api_key or config.api_key in ("sk-your-api-key-here", "your-api-key"):
        print("错误: 请先配置 LLM_API_KEY")
        print("复制 .env.example 为 .env，设置 LLM_PROVIDER 和 LLM_API_KEY")
        sys.exit(1)

    log = _get_logger()
    log.info("Minimal Agent 启动: model=%s max_steps=%d", config.model, config.max_steps)

    print("=" * 50)
    print("  Minimal Agent v0.1.0")
    print(f"  Provider: {config.provider}")
    print(f"  Model: {config.model}")
    print(f"  Base URL: {config.base_url}")
    print(f"  Max Steps: {config.max_steps}")
    print(f"  Trace: logs/agent.log")
    print("=" * 50)

    try:
        asyncio.run(_run_interactive(config))
    except KeyboardInterrupt:
        print("\n  再见！")
    finally:
        log.info("Minimal Agent 退出")


if __name__ == "__main__":
    main()
