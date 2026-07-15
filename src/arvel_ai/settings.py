"""Typed, validated views over the ``ai`` config section — the framework's
``Settings`` pattern (msgspec, auto-loading, coercing), same shape as
``MailSettings`` / ``FilesystemSettings``.

``AiSettings()`` reads + validates ``config("ai")``; field defaults ARE the
package defaults (no separate DEFAULTS dict, no ``merge_config_from``). API keys
never live here — driver blocks name the ENV VAR holding the key.
"""

from __future__ import annotations

from typing import Any

import msgspec

from arvel.kernel.settings import Settings


class OpenAICompatibleSettings(msgspec.Struct):
    base_url: str | None = None
    api_key_env: str = "AI_API_KEY"  # NAME of the env var holding the key
    model: str | None = None
    timeout: float = 60.0
    include_raw: bool = False


class LiteLLMSettings(msgspec.Struct):
    model: str | None = None
    timeout: float = 60.0
    max_retries: int = 2
    include_raw: bool = False


class DriverSettings(msgspec.Struct):
    litellm: LiteLLMSettings = msgspec.field(default_factory=LiteLLMSettings)
    openai_compatible: OpenAICompatibleSettings = msgspec.field(
        default_factory=OpenAICompatibleSettings
    )


class McpAuthSettings(msgspec.Struct):
    mode: str = "token"  # "token" | "oidc"
    token_env: str = "MCP_TOKEN"  # NAME of the env var holding the bearer token
    issuer: str | None = None
    jwks_uri: str | None = None
    audience: str | None = None


class McpSettings(msgspec.Struct):
    enabled: bool = False  # off by default — exposing the app to agents is deliberate
    path: str = "/mcp"
    public_url: str | None = None  # canonical https URL (required when enabled)
    tools: list[str] = msgspec.field(default_factory=list)  # modules to import at boot
    auth: McpAuthSettings = msgspec.field(default_factory=McpAuthSettings)


class TemporalSettings(msgspec.Struct):
    target: str = "localhost:7233"
    namespace: str = "default"
    task_queue: str = "arvel-ai"


class WorkflowDriverSettings(msgspec.Struct):
    temporal: TemporalSettings = msgspec.field(default_factory=TemporalSettings)


class WorkflowSettings(msgspec.Struct):
    default: str = "queue"  # "queue" | "temporal" | "fake"
    drivers: WorkflowDriverSettings = msgspec.field(default_factory=WorkflowDriverSettings)


class AiSettings(Settings):
    """Typed view over ``config("ai")`` — auto-loads + validates on instantiation."""

    __config_key__ = "ai"

    default: str = "litellm"  # which driver AI.chat() dispatches to
    # model aliases so callers say AI.chat(..., model="fast") and ops map it to a real id here
    models: dict[str, str] = msgspec.field(default_factory=dict)
    drivers: DriverSettings = msgspec.field(default_factory=DriverSettings)
    include_raw: bool = False
    critical: bool = False  # AiResource: does an AI outage abort boot?
    mcp: McpSettings = msgspec.field(default_factory=McpSettings)
    workflows: WorkflowSettings = msgspec.field(default_factory=WorkflowSettings)


def _as_kwargs(struct: msgspec.Struct) -> dict[str, Any]:
    """A driver settings struct → kwargs for the driver constructor."""
    return {f: getattr(struct, f) for f in struct.__struct_fields__}
