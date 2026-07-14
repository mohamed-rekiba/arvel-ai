"""LiteLLM SDK driver — 100+ providers behind the stable contract (DR-0041).

LiteLLM normalizes providers to the OpenAI format, so translation is shared
with the openai_compatible driver. litellm is confined to this module (the
import-linter contract enforces it), lazy-imported, and installed via
`uv add 'arvel-ai[litellm]'`. Provider keys come from each provider's own env
var (ANTHROPIC_API_KEY, OPENAI_API_KEY, ...).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from arvel.support.manager import MissingExtraError

from arvel_ai.contracts import (
    AiAuthError,
    AiContentFiltered,
    AiError,
    AiInvalidRequest,
    AiProviderError,
    AiRateLimited,
    AiTimeout,
    ChatDelta,
    ChatRequest,
    ChatResponse,
    EmbedRequest,
    EmbedResponse,
    StreamEnd,
    Usage,
)

from ._openai_format import parse_openai_response, to_openai_payload


class LiteLLMDriver:
    supports_embeddings = True

    def __init__(
        self,
        model: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        include_raw: bool = False,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.include_raw = include_raw

    def _litellm(self) -> Any:
        try:
            import litellm
        except ImportError as exc:
            raise MissingExtraError("litellm", package="arvel-ai") from exc
        return litellm

    async def chat(self, request: ChatRequest) -> ChatResponse:
        litellm = self._litellm()
        payload = to_openai_payload(request, self.model)
        try:
            result = await litellm.acompletion(
                **payload, timeout=self.timeout, num_retries=self.max_retries
            )
        except Exception as exc:  # noqa: BLE001 - translated at the boundary
            raise self._translate(exc) from exc
        return parse_openai_response(
            result.model_dump() if hasattr(result, "model_dump") else dict(result),
            include_raw=self.include_raw,
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatDelta]:
        # ponytail: v1 streams via one buffered completion per S1 (text deltas
        # from litellm chunks; tool calls buffered). Upgrade: wire litellm's
        # chunk stream through when the openai_compatible SSE path stabilizes.
        litellm = self._litellm()
        payload = to_openai_payload(request, self.model)
        try:
            stream = await litellm.acompletion(
                **payload, stream=True, timeout=self.timeout, num_retries=self.max_retries
            )
            text_parts: list[str] = []
            model = ""
            finish = "stop"
            async for chunk in stream:
                data = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
                model = data.get("model") or model
                choice = (data.get("choices") or [{}])[0]
                finish = choice.get("finish_reason") or finish
                content = (choice.get("delta") or {}).get("content")
                if content:
                    text_parts.append(content)
                    from arvel_ai.contracts import TextDelta

                    yield TextDelta(text=content)
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc) from exc
        from arvel_ai.contracts import Text

        from ._openai_format import _FINISH_REASONS  # noqa: PLC0415

        yield StreamEnd(
            response=ChatResponse(
                content=[Text(text="".join(text_parts))] if text_parts else [],
                stop_reason=_FINISH_REASONS.get(finish, "other"),
                model=model,
            )
        )

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        litellm = self._litellm()
        try:
            result = await litellm.aembedding(
                model=request.model or self.model, input=request.texts, timeout=self.timeout
            )
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc) from exc
        data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        usage = data.get("usage") or {}
        return EmbedResponse(
            vectors=[item["embedding"] for item in data.get("data", [])],
            model=data.get("model", ""),
            usage=Usage(input_tokens=usage.get("prompt_tokens", 0)),
        )

    # -- error mapping: litellm exceptions -> the S1 taxonomy --------------------

    @staticmethod
    def _translate(exc: Exception) -> AiError:
        if isinstance(exc, AiError):
            return exc
        name = type(exc).__name__
        status = getattr(exc, "status_code", None)
        message = f"{name}: {exc}"
        if name in ("AuthenticationError", "PermissionDeniedError") or status in (401, 403):
            return AiAuthError(message)
        if name == "RateLimitError" or status == 429:
            return AiRateLimited(message)
        if name in ("Timeout", "APITimeoutError"):
            return AiTimeout(message)
        if name == "ContentPolicyViolationError":
            return AiContentFiltered(message)
        if name in ("BadRequestError", "NotFoundError", "UnprocessableEntityError") or (
            status is not None and 400 <= int(status) < 500
        ):
            return AiInvalidRequest(message)
        return AiProviderError(message)
