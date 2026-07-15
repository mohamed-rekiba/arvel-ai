"""The fake driver — a first-class driver, not a bolt-on test helper.

`AI.fake()` swaps it in (mirroring `Mail::fake()`); apps script replies and
assert on recorded requests. It is also this package's own red-green harness.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, TypeVar

from arvel_ai.contracts import (
    ChatDelta,
    ChatRequest,
    ChatResponse,
    EmbedRequest,
    EmbedResponse,
    Message,
    StreamEnd,
    Text,
    TextDelta,
    Usage,
)

T = TypeVar("T")


def _message_text(message: Message) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(p.text for p in message.content if isinstance(p, Text))


class FakeAiDriver:
    supports_embeddings = True

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies: list[str] = list(replies or ["ok"])
        self.requests: list[ChatRequest] = []
        self.embedded: list[list[str]] = []

    # -- scripting ----------------------------------------------------------

    def _next_reply(self) -> str:
        if len(self.replies) > 1:
            return self.replies.pop(0)
        return self.replies[0]  # the last scripted reply sticks

    # -- driver contract ------------------------------------------------------

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        reply = self._next_reply()
        return ChatResponse(
            content=[Text(text=reply)],
            stop_reason="end_turn",
            model="fake",
            usage=Usage(input_tokens=0, output_tokens=0),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatDelta]:
        response = await self.chat(request)
        for char in response.text:
            yield TextDelta(text=char)
        yield StreamEnd(response=response)

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        self.embedded.append(list(request.texts))
        return EmbedResponse(
            vectors=[[float(len(t)), 0.0, 1.0] for t in request.texts], model="fake"
        )

    # convenience mirroring manager sugar, so a faked facade keeps the same verbs
    async def embed_texts(self, texts: list[str], model: str | None = None) -> EmbedResponse:
        return await self.embed(EmbedRequest(texts=texts, model=model))

    async def structured(self, schema: type[T], messages: object, **kwargs: Any) -> T:
        request = ChatRequest(
            messages=[Message(role="user", content=str(messages))],
            response_schema=schema,
            model=kwargs.get("model"),
            system=kwargs.get("system"),
        )
        response = await self.chat(request)
        return response.structured(schema)

    # -- assertions -----------------------------------------------------------

    def assert_chatted(self, fragment: str) -> None:
        for request in self.requests:
            if any(fragment in _message_text(m) for m in request.messages):
                return
        raise AssertionError(f"no chat request contained {fragment!r}")
