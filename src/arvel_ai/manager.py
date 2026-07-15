"""AiManager — config-selected driver dispatch plus the caller-facing sugar.

`AI.chat("...")` / `stream` / `structured` / `embed` land here: the manager
turns ergonomic arguments into a frozen ChatRequest (str sugar, model-alias
resolution) and delegates to the configured driver. `.extend()` is the seam
for custom drivers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

from arvel.support.manager import Manager

from .contracts import (
    AiCapabilityError,
    AiDriver,
    ChatDelta,
    ChatRequest,
    ChatResponse,
    EmbedRequest,
    EmbedResponse,
    Message,
    StreamEnd,
    ToolDef,
)
from .events import AiEmbedding, AiRequestFailed, AiRequestSending, AiResponseReceived
from .settings import AiSettings, _as_kwargs

MessagesInput = str | list[Message] | ChatRequest


class AiManager(Manager):
    # MissingExtraError hints name this distribution, not arvel core.
    extra_package = "arvel-ai"

    # -- driver wiring (typed Settings — the framework's config pattern) -------

    def settings(self) -> AiSettings:
        return self._settings(AiSettings)

    def default_driver(self) -> str:
        return self.settings().default

    def create_fake_driver(self) -> Any:
        from .drivers.fake import FakeAiDriver

        return FakeAiDriver()

    def create_openai_compatible_driver(self) -> Any:
        from .drivers.openai_compatible import OpenAICompatibleDriver

        return OpenAICompatibleDriver(**_as_kwargs(self.settings().drivers.openai_compatible))

    def create_litellm_driver(self) -> Any:
        from .drivers.litellm import LiteLLMDriver

        return LiteLLMDriver(**_as_kwargs(self.settings().drivers.litellm))

    # -- request building -------------------------------------------------------

    def resolve_model(self, model: str | None) -> str | None:
        """Alias -> concrete id via config `ai.models` (the model-churn shield):
        apps say "fast"/"smart"; a provider retiring a model is a config edit."""
        if model is None:
            return None
        return self.settings().models.get(model, model)

    def _build_request(
        self,
        messages: MessagesInput,
        *,
        model: str | None = None,
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        tool_choice: str = "auto",
        response_schema: Any = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatRequest:
        if isinstance(messages, ChatRequest):
            request = messages
            request.model = self.resolve_model(request.model)
            return request
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return ChatRequest(
            messages=messages,
            model=self.resolve_model(model),
            system=system,
            tools=tools or [],
            tool_choice=tool_choice,
            response_schema=response_schema,
            max_tokens=max_tokens,
            stop=stop or [],
            options=options or {},
        )

    # -- caller-facing verbs -----------------------------------------------------

    def _driver(self) -> AiDriver:
        return cast(AiDriver, self.driver())

    # -- observability (arvel.telemetry span is a no-op when tracing is off) ----

    async def _dispatch(self, event: Any) -> None:
        if self.app is not None and self.app.bound("events"):
            await self.app.make("events").dispatch(event)

    def _span_attributes(self, request: ChatRequest) -> dict[str, Any]:
        return {
            "ai.driver": self.default_driver(),
            "ai.model": request.model or "",
        }

    async def chat(self, messages: MessagesInput, **kwargs: Any) -> ChatResponse:
        from arvel.telemetry import span

        request = self._build_request(messages, **kwargs)
        await self._dispatch(AiRequestSending(self.default_driver(), request))
        with span("ai.chat", kind="client", attributes=self._span_attributes(request)) as current:
            try:
                response = await self._driver().chat(request)
            except Exception as exc:
                await self._dispatch(AiRequestFailed(self.default_driver(), request, exc))
                raise
            if current is not None:
                current.set_attribute("ai.input_tokens", response.usage.input_tokens)
                current.set_attribute("ai.output_tokens", response.usage.output_tokens)
        await self._dispatch(AiResponseReceived(self.default_driver(), request, response))
        return response

    async def stream(self, messages: MessagesInput, **kwargs: Any) -> AsyncIterator[ChatDelta]:
        from arvel.telemetry import span

        request = self._build_request(messages, **kwargs)
        await self._dispatch(AiRequestSending(self.default_driver(), request))
        with span("ai.stream", kind="client", attributes=self._span_attributes(request)):
            async for delta in self._driver().stream(request):
                if isinstance(delta, StreamEnd):
                    await self._dispatch(
                        AiResponseReceived(self.default_driver(), request, delta.response)
                    )
                yield delta

    async def structured(self, schema: type, messages: MessagesInput, **kwargs: Any) -> Any:
        response = await self.chat(messages, response_schema=schema, **kwargs)
        return response.structured(schema)

    async def embed(self, texts: list[str], model: str | None = None) -> EmbedResponse:
        from arvel.telemetry import span

        driver = self._driver()
        if not getattr(driver, "supports_embeddings", False):
            raise AiCapabilityError(
                f"driver {self.default_driver()!r} has no embeddings support - "
                "configure a driver that does (e.g. litellm/openai_compatible)"
            )
        request = EmbedRequest(texts=texts, model=self.resolve_model(model))
        await self._dispatch(AiEmbedding(self.default_driver(), request))
        with span(
            "ai.embed", kind="client", attributes={"ai.driver": self.default_driver()}
        ):
            return await driver.embed(request)
