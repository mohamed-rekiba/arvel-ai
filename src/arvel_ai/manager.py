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
    ToolDef,
)

MessagesInput = str | list[Message] | ChatRequest


class AiManager(Manager):
    # MissingExtraError hints name this distribution, not arvel core.
    extra_package = "arvel-ai"

    # -- driver wiring --------------------------------------------------------

    def _config(self, key: str, default: Any = None) -> Any:
        if self.app is None:
            return default
        return self.app.make("config").get(f"ai.{key}", default)

    def default_driver(self) -> str:
        return str(self._config("default", "litellm"))

    def create_fake_driver(self) -> Any:
        from .drivers.fake import FakeAiDriver

        return FakeAiDriver()

    def create_openai_compatible_driver(self) -> Any:
        from .drivers.openai_compatible import OpenAICompatibleDriver

        return OpenAICompatibleDriver(**(self._config("drivers.openai_compatible", {}) or {}))

    def create_litellm_driver(self) -> Any:
        from .drivers.litellm import LiteLLMDriver

        return LiteLLMDriver(**(self._config("drivers.litellm", {}) or {}))

    # -- request building -------------------------------------------------------

    def resolve_model(self, model: str | None) -> str | None:
        """Alias -> concrete id via config `ai.models` (the model-churn shield):
        apps say "fast"/"smart"; a provider retiring a model is a config edit."""
        if model is None:
            return None
        aliases: dict[str, str] = self._config("models", {}) or {}
        return aliases.get(model, model)

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

    async def chat(self, messages: MessagesInput, **kwargs: Any) -> ChatResponse:
        return await self._driver().chat(self._build_request(messages, **kwargs))

    def stream(self, messages: MessagesInput, **kwargs: Any) -> AsyncIterator[ChatDelta]:
        return self._driver().stream(self._build_request(messages, **kwargs))

    async def structured(self, schema: type, messages: MessagesInput, **kwargs: Any) -> Any:
        request = self._build_request(messages, response_schema=schema, **kwargs)
        response = await self._driver().chat(request)
        return response.structured(schema)

    async def embed(self, texts: list[str], model: str | None = None) -> EmbedResponse:
        driver = self._driver()
        if not getattr(driver, "supports_embeddings", False):
            raise AiCapabilityError(
                f"driver {self.default_driver()!r} has no embeddings support - "
                "configure a driver that does (e.g. litellm/openai_compatible)"
            )
        return await driver.embed(EmbedRequest(texts=texts, model=self.resolve_model(model)))
