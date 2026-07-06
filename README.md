# Minimal Agent

从零实现的最小可用 AI Agent Runtime —— 一个轻量级的本地 AI Agent 系统。

## 快速开始

### 环境配置

```bash
# 1. 复制配置文件
cp .env.example .env

# 2. 编辑 .env，填入你的 OpenAI 兼容 API Key
# OPENAI_API_KEY=sk-your-api-key
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_MODEL=gpt-4o-mini

# 3. 安装依赖
uv sync

# 4. 运行
uv run minimal-agent
```

### CLI 命令

| 命令 | 说明 |
|------|------|
| 直接输入文本 | 向当前 session 发送消息 |
| `/new [标题]` | 创建新 session |
| `/switch <id>` | 切换到指定 session |
| `/sessions` | 列出所有 session |
| `/close` | 关闭当前 session |
| `/help` | 显示帮助 |
| `/trace` | 显示日志路径 |
| `/exit` | 退出 |

### 运行测试

```bash
uv run pytest tests/ -v
```

## 系统设计

### 架构总览

```
CLI (main.py)
  ├── Config         ← .env 文件加载配置
  ├── SessionManager ← 多 session 管理（独立对话上下文）
  │   └── AgentLoop  ← plan → act → observe 核心循环
  │        ├── LLMProvider    ← OpenAI 兼容 API 调用
  │        ├── ToolRegistry   ← 工具注册与分发
  │        │    ├── CalculatorTool
  │        │    ├── SearchTool
  │        │    ├── WeatherTool
  │        │    ├── TodoAddTool / TodoListTool / TodoDoneTool
  │        └── ExecutionContext ← 状态管理 + 上下文压缩
  └── Trace (logs/agent.log) ← 完整执行日志
```

### 核心循环 (AgentLoop)

```
while not context.is_done():
    1. [plan] 调用 LLM，传入 messages + tool_schemas
    2. [observe] 解析 LLM 响应中的 text / tool_use
    3. [act] 执行工具调用，结果追加到 context
    4. 检查终止条件 (end_turn / max_steps)
    5. 上下文水位检测 → 触达阈值则压缩
```

### 工具系统

每个工具继承 `BaseTool`，定义三个要素：

| 要素 | 说明 |
|------|------|
| `name` | 工具名称，LLM 据此调用 |
| `description` | 工具描述，LLM 据此决策 |
| `input_schema` | JSON Schema 格式参数定义 |

LLM 基于 Function Calling 机制自主决策调用哪个工具。运行时通过 `ToolRegistry` 注册和查找。

#### 已实现工具

| 工具 | 用途 |
|------|------|
| `calculator` | 安全数学表达式求值（白名单过滤，禁止任意代码执行） |
| `search` | 模拟网络搜索（预置 mock 数据） |
| `weather` | 模拟天气查询（随机生成数据） |
| `todo_add/list/done` | 待办事项管理（内存存储，支持增/查/改） |

### Session 管理

每个 session 拥有独立的：
- **对话历史** (`messages`)：完整的多轮对话记录
- **执行锁** (`asyncio.Lock`)：保证同一 session 串行执行
- **状态标记** (`active / closed`)

用户 A 可以同时打开窗口 1（查天气 + 记待办）和窗口 2（写周报 + 记待办），两个窗口的对话历史和上下文完全隔离，互不干扰。

### Context 管理

#### Memory 的存放方式

每次 run 的上下文由以下部分组成，按顺序注入 system prompt：

```
system prompt（工具使用指引）
  → 对话历史（session.messages 完整回放）
    → user: 用户输入
    → assistant: [thinking, text, tool_use]
    → tool: 工具执行结果
  → ...（多轮循环）
```

#### Memory 的召回时机

- **追问**：每次发送新消息时，session 完整消息历史作为 `prefill_messages` 注入新的 `ExecutionContext`
- **跨 session**：不共享记忆，每个 session 独立维护自己的 `messages` 列表
- **工具结果**：工具返回内容自动追加到 conversation，LLM 在下一轮可看到

#### 上下文压缩

- **触发条件**：估算 token 数 > `max_context_tokens × compact_threshold`（默认 8000 × 0.75 = 6000）
- **压缩策略**：保留首条用户消息 + 最近 4 轮，中间部分调用 LLM 生成摘要替换
- **摘要格式**：`[对话摘要] + 简短确认消息 + 最近 N 轮`

### Trace / 日志

所有执行日志写入 `logs/agent.log`，包含：
- 工具调用开始/完成（工具名、参数、耗时、错误）
- Run 完成状态（steps、reason）
- 异常堆栈

终端交互过程中实时显示：
- 每轮 step 编号
- 工具调用参数和结果
- 上下文 token 估算

## 项目结构

```
minimal_agent/
├── pyproject.toml               # 项目配置
├── .env.example                 # 环境变量模板
├── src/minimal_agent/
│   ├── __init__.py
│   ├── config.py                # 配置加载（.env → 环境变量）
│   ├── llm.py                   # LLM Provider（OpenAI 兼容 API）
│   ├── loop.py                  # AgentLoop（plan→act→observe）
│   ├── context.py               # ExecutionContext + 压缩
│   ├── session.py               # Session 管理（多会话隔离）
│   ├── main.py                  # CLI 入口
│   └── tools/
│       ├── __init__.py
│       ├── base.py              # BaseTool 抽象类
│       ├── registry.py          # ToolRegistry
│       ├── calculator.py        # 计算器
│       ├── search.py            # 搜索（mock）
│       ├── weather.py           # 天气（mock）
│       └── todo.py              # 待办事项
└── tests/
    ├── test_tools.py            # 工具单元测试
    ├── test_context.py          # 上下文管理测试
    ├── test_session.py          # Session 隔离测试
    └── test_loop.py             # AgentLoop 集成测试
```

## 技术要点

- **从零实现**：不依赖任何 Agent 框架（langgraph/openhands/openclaw），核心 Runtime 完全自研
- **LLM 输出解析**：支持 text、thinking、tool_use 三种内容块的解析
- **异常处理**：工具超时（60s）、未知工具、LLM 调用失败、max_steps 超限
- **错误恢复**：工具执行异常不中断循环，错误作为 tool_result 返回给 LLM 自主处理
