"""Type stub for the ``AI`` facade — restores static completion + type-safety on a surface
that is otherwise opaque to type-checkers (``Facade.__getattr__`` proxies to the AiManager).
Hand-maintained to mirror ``AiManager``'s public methods (chat/stream/structured/embed)."""

from collections.abc import AsyncIterator
from typing import Any

from arvel.support.facades import Facade

from .contracts import ChatDelta, ChatRequest, ChatResponse, EmbedResponse, Message
from .drivers.fake import FakeAiDriver

MessagesInput = str | list[Message] | ChatRequest

class AI(Facade):
    @classmethod
    async def chat(cls, messages: MessagesInput, **kwargs: Any) -> ChatResponse: ...
    @classmethod
    def stream(cls, messages: MessagesInput, **kwargs: Any) -> AsyncIterator[ChatDelta]: ...
    @classmethod
    async def structured[T](cls, schema: type[T], messages: MessagesInput, **kwargs: Any) -> T: ...
    @classmethod
    async def embed(cls, texts: list[str], model: str | None = None) -> EmbedResponse: ...
    @classmethod
    def fake(cls) -> FakeAiDriver: ...
