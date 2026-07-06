# 配置加载：从 .env 文件和环境变量读取所有配置项，支持多 LLM Provider 预设
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Provider 预设：自动填充 base_url 和默认模型
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "default_model": "claude-sonnet-4-20250514",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
    },
    "xiaomi": {
        "base_url": "https://api.xiaomimlm.com/v1",
        "default_model": "mixtral-8x7b",
    },
    "custom": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
}


@dataclass
class Config:
    provider: str = "openai"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    extra_headers: dict[str, str] = field(default_factory=dict)
    max_steps: int = 20
    max_context_tokens: int = 8000
    compact_threshold: float = 0.75

    @property
    def is_anthropic(self) -> bool:
        return self.provider == "anthropic"

    @property
    def is_openai_compatible(self) -> bool:
        return self.provider in ("openai", "deepseek", "xiaomi", "custom")

    def resolve_model_max_tokens(self) -> int:
        model_lower = self.model.lower()
        # Anthropic Claude 系列 — 200K context
        if any(k in model_lower for k in ("claude",)):
            return 200000
        if any(k in model_lower for k in ("gpt-4", "deepseek", "r1", "v3")):
            return 128000
        if "32k" in model_lower:
            return 32000
        return 8000


# 从环境变量加载，支持 LLM_PROVIDER 快速切换
def load_config(env_file: str | None = None) -> Config:
    if env_file:
        load_dotenv(env_file, override=True)
    else:
        for candidate in [Path(".env"), Path(__file__).resolve().parents[3] / ".env"]:
            if candidate.exists():
                load_dotenv(str(candidate), override=True)
                break
        else:
            load_dotenv(override=True)

    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["openai"])

    base_url = os.getenv("LLM_BASE_URL", "") or preset["base_url"]
    model = os.getenv("LLM_MODEL", "") or preset["default_model"]
    api_key = os.getenv("LLM_API_KEY", "") or os.getenv("OPENAI_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "")

    extra_headers: dict[str, str] = {}
    raw_headers = os.getenv("LLM_EXTRA_HEADERS", "")
    if raw_headers:
        try:
            extra_headers = json.loads(raw_headers)
        except json.JSONDecodeError:
            pass

    return Config(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        extra_headers=extra_headers,
        max_steps=int(os.getenv("AGENT_MAX_STEPS", "20")),
        max_context_tokens=int(os.getenv("AGENT_MAX_CONTEXT_TOKENS", "8000")),
        compact_threshold=float(os.getenv("AGENT_COMPACT_THRESHOLD", "0.75")),
    )
