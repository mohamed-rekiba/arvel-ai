"""LiteLLM driver — one contract in front of 100+ providers.

LiteLLM normalizes every provider to the OpenAI format, so the request/response translation is
the same code the openai_compatible driver uses. The litellm import lives only in this module
and is loaded lazily, so an app that doesn't use it never pays for it; install it with
`uv add 'arvel-ai[litellm]'`. Each provider's key comes from its own env var
(ANTHROPIC_API_KEY, OPENAI_API_KEY, and so on).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from arvel.contracts import HealthResult, HealthStatus
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
    Text,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    Usage,
)

from ._openai_format import (
    _FINISH_REASONS,
    decode_tool_arguments,
    parse_openai_response,
    to_openai_payload,
)

_HEALTH_TIMEOUT = 5.0  # keep the boot/health probe snappy — don't hang startup on a slow provider
# transport/auth-level failures mean the provider is unusable; a request-level error (bad request,
# rate limit, content filter) means it's reachable + authed but impaired
_HEALTH_FAILED = (AiAuthError, AiTimeout, AiProviderError)


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

    async def health(self) -> HealthResult:
        """Check the provider by actually asking for a one-token completion, so it goes through
        the same auth and provider path a real call would. A bad or missing key -> failed, an
        unreachable or slow provider -> failed, a request-level error -> degraded, and a real
        completion -> ok. Non-critical, so a failure degrades startup rather than aborting it."""
        if not self.model:
            return HealthResult(HealthStatus.DEGRADED, detail="no model configured")
        try:
            litellm = self._litellm()
        except MissingExtraError as exc:
            return HealthResult(HealthStatus.FAILED, detail=str(exc))
        try:
            await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                timeout=_HEALTH_TIMEOUT,
                num_retries=0,
            )
        except Exception as exc:  # noqa: BLE001 - translated at the boundary
            err = self._translate(exc)
            status = (
                HealthStatus.FAILED if isinstance(err, _HEALTH_FAILED) else HealthStatus.DEGRADED
            )
            return HealthResult(status, detail=f"{type(err).__name__}: {err}")
        return HealthResult(HealthStatus.OK, detail="completion reachable")

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
        litellm = self._litellm()
        payload = to_openai_payload(request, self.model)
        text_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        model = ""
        finish = "stop"
        try:
            stream = await litellm.acompletion(
                **payload, stream=True, timeout=self.timeout, num_retries=self.max_retries
            )
            async for chunk in stream:
                data = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
                model = data.get("model") or model
                choice = (data.get("choices") or [{}])[0]
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    text_parts.append(content)
                    yield TextDelta(text=content)
                for tc in delta.get("tool_calls") or []:
                    index = tc.get("index", 0)
                    slot = tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    tc_id, name = tc.get("id"), fn.get("name")
                    args_fragment = fn.get("arguments") or ""
                    slot["id"] = tc_id or slot["id"]
                    slot["name"] = name or slot["name"]
                    slot["arguments"] += args_fragment
                    yield ToolCallDelta(index=index, id=tc_id, name=name, arguments=args_fragment)
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc) from exc
        content_parts: list[Any] = [Text(text="".join(text_parts))] if text_parts else []
        for slot in tool_calls.values():
            content_parts.append(
                ToolCall(
                    id=slot["id"],
                    name=slot["name"],
                    arguments=decode_tool_arguments(slot["arguments"]),
                )
            )
        yield StreamEnd(
            response=ChatResponse(
                content=content_parts,
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

    # -- turning litellm exceptions into AiErrors --------------------------------

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
