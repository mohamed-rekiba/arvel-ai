"""Package config defaults, merged under the "ai" key.

The host app overrides any of these in its own config/ai.py — app values always
win. API keys come from ENV VARS only (each driver documents which); never put
a key in config.
"""

from __future__ import annotations

from typing import Any

DEFAULTS: dict[str, Any] = {
    # which driver AI.chat() dispatches to; swap providers here, not in code
    "default": "litellm",
    # model aliases — the churn shield. Apps say AI.chat(..., model="fast");
    # a provider retiring a model is one config edit, zero code changes.
    "models": {
        # "fast": "claude-haiku-4-5",
        # "smart": "claude-opus-4-8",
    },
    "drivers": {
        # LiteLLM SDK engine (uv add 'arvel-ai[litellm]') — 100+ providers.
        # Keys via each provider's own env var (ANTHROPIC_API_KEY, ...).
        "litellm": {
            "model": None,  # default model when a request names none
            "timeout": 60.0,
            "max_retries": 2,
        },
        # Any OpenAI-format endpoint: a deployed LiteLLM proxy, vLLM, Ollama's
        # OpenAI route, ... (uv add 'arvel-ai[httpx]')
        "openai_compatible": {
            "base_url": None,
            "api_key_env": "AI_API_KEY",  # NAME of the env var holding the key
            "model": None,
            "timeout": 60.0,
        },
        "fake": {},
    },
    # opt-in: attach the provider-native payload to ChatResponse.raw
    "include_raw": False,
}
