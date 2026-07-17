"""The gateway's stable shapes — all arvel-owned, no provider or engine types.

Each driver translates between these models and its provider's wire format at the edge, so
nothing from any_llm, httpx, or a provider SDK ever reaches this module. The import-linter
rule in pyproject.toml keeps it that way.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

import msgspec

T = TypeVar("T")

Role = Literal["user", "assistant"]
StopReason = Literal["end_turn", "max_tokens", "tool_use", "refusal", "other"]


# ---- content parts ---------------------------------------------------------


class Text(msgspec.Struct, tag="text"):
    text: str


class Image(msgspec.Struct, tag="image"):
    media_type: str
    data: str | None = None  # base64
    url: str | None = None


class ToolCall(msgspec.Struct, tag="tool_call"):
    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(msgspec.Struct, tag="tool_result"):
    tool_call_id: str
    content: str
    is_error: bool = False


ContentPart = Text | Image | ToolCall | ToolResult


class Message(msgspec.Struct):
    role: Role
    content: str | list[ContentPart]  # plain str == one Text part


# ---- requests --------------------------------------------------------------


class ToolDef(msgspec.Struct):
    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema (stick to the basics every provider accepts)


class ChatRequest(msgspec.Struct):
    messages: list[Message]
    model: str | None = None  # concrete id (aliases resolve in the manager)
    system: str | None = None
    tools: list[ToolDef] = msgspec.field(default_factory=list[ToolDef])
    tool_choice: str = "auto"  # "auto" | "none" | "required" | <tool name>
    response_schema: Any = None  # type[msgspec.Struct] | JSON-schema dict
    max_tokens: int | None = None
    stop: list[str] = msgspec.field(default_factory=list[str])
    options: dict[str, Any] = msgspec.field(default_factory=dict[str, Any])  # provider passthrough


class EmbedRequest(msgspec.Struct):
    texts: list[str]
    model: str | None = None


# ---- responses -------------------------------------------------------------


class Usage(msgspec.Struct):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0  # 0 where the provider doesn't report it


class ChatResponse(msgspec.Struct):
    content: list[ContentPart]
    stop_reason: StopReason = "end_turn"
    model: str = ""
    usage: Usage = msgspec.field(default_factory=Usage)
    raw: dict[str, Any] | None = None  # provider-native response, opt-in

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.content if isinstance(p, Text))

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [p for p in self.content if isinstance(p, ToolCall)]

    def structured(self, schema: type[T]) -> T:
        """Decode the text content into ``schema`` (a msgspec Struct type) —
        generic, so the caller gets a typed instance back, not ``Any``."""
        return msgspec.json.decode(self.text.encode(), type=schema)


class EmbedResponse(msgspec.Struct):
    vectors: list[list[float]]
    model: str = ""
    usage: Usage = msgspec.field(default_factory=Usage)


# ---- streaming -------------------------------------------------------------


class TextDelta(msgspec.Struct, tag="delta"):
    text: str


class ToolCallDelta(msgspec.Struct, tag="tool_call_delta"):
    """An incremental tool-call fragment during streaming. Providers emit a call's
    ``id``/``name`` once (early) and its ``arguments`` as a JSON string split across
    deltas — concatenate ``arguments`` per ``index`` to rebuild the call. The complete
    calls are still buffered into ``StreamEnd.response.tool_calls``, so a consumer that
    doesn't need token-level tool streaming can ignore these and read the end."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""  # a fragment of the arguments JSON string (empty for the opening delta)


class StreamEnd(msgspec.Struct, tag="end"):
    response: ChatResponse


ChatDelta = TextDelta | ToolCallDelta | StreamEnd


# ---- errors -----------------------------------------------------------------


class AiError(Exception):
    """Base for every gateway failure. ``retryable`` says whether backoff+retry
    is sane; drivers translate provider exceptions into exactly one of these."""

    retryable = False


class AiAuthError(AiError):
    pass


class AiInvalidRequest(AiError):
    pass


class AiRateLimited(AiError):
    retryable = True

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class AiProviderError(AiError):
    retryable = True


class AiTimeout(AiError):
    retryable = True


class AiContentFiltered(AiError):
    """A pre-output refusal/content-filter outcome, raised by drivers. A refusal
    that carries partial content surfaces as stop_reason='refusal' instead."""


class AiCapabilityError(AiError):
    """The selected driver can't do this (e.g. embeddings on a provider
    without an embeddings endpoint)."""


# ---- the driver contract ----------------------------------------------------


@runtime_checkable
class AiDriver(Protocol):
    """What every AI driver implements. Three methods is the whole surface:
    structured output and tools ride on ChatRequest fields."""

    supports_embeddings: bool

    async def chat(self, request: ChatRequest) -> ChatResponse: ...

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatDelta]: ...

    async def embed(self, request: EmbedRequest) -> EmbedResponse: ...
