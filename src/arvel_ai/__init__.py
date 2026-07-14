"""arvel-ai — one stable API over many AI providers, plus an MCP server.

Quickstart:

    from arvel_ai import AI

    reply = await AI.chat("Summarize this product", model="fast")
    async for delta in AI.stream("Write a description"): ...
    product = await AI.structured(ProductCopy, "Generate copy for ...")
"""

from .contracts import (
    AiAuthError,
    AiCapabilityError,
    AiContentFiltered,
    AiError,
    AiInvalidRequest,
    AiProviderError,
    AiRateLimited,
    AiTimeout,
    ChatRequest,
    ChatResponse,
    EmbedResponse,
    Message,
    StreamEnd,
    Text,
    TextDelta,
    ToolCall,
    ToolDef,
    ToolResult,
    Usage,
)
from .facade import AI

__all__ = [
    "AI",
    "AiAuthError",
    "AiCapabilityError",
    "AiContentFiltered",
    "AiError",
    "AiInvalidRequest",
    "AiProviderError",
    "AiRateLimited",
    "AiTimeout",
    "ChatRequest",
    "ChatResponse",
    "EmbedResponse",
    "Message",
    "StreamEnd",
    "Text",
    "TextDelta",
    "ToolCall",
    "ToolDef",
    "ToolResult",
    "Usage",
]
