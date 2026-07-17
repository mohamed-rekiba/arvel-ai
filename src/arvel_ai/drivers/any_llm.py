"""any-llm driver — one contract in front of many providers.

any-llm (Mozilla AI) normalizes every provider to the OpenAI format, so the request/response
translation is the same code the openai_compatible driver uses. The any_llm import lives only
in this module and is loaded lazily, so an app that doesn't use it never pays for it; one
extra installs the driver plus your provider's SDK — `uv add 'arvel-ai[anthropic]'` (arvel-ai
mirrors any-llm's provider extras; `any-llm` is the bare SDK, `all` every provider). Model
ids are `provider:model` colon-separated (`anthropic:claude-haiku-4-5`), and each provider's
key comes from its own env var (ANTHROPIC_API_KEY, OPENAI_API_KEY, and so on).

any-llm exposes no timeout or retry knobs, so both live here: every call runs under
`asyncio.wait_for` (per-chunk while streaming), and chat/embed retry retryable failures
with exponential backoff. Streaming never retries — deltas already yielded can't be unsaid.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
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
    FINISH_REASONS,
    decode_tool_arguments,
    parse_openai_response,
    to_openai_payload,
)

_HEALTH_TIMEOUT = 5.0  # keep the boot/health probe snappy — don't hang startup on a slow provider
# transport/auth-level failures mean the provider is unusable; a request-level error (bad request,
# rate limit, content filter) means it's reachable + authed but impaired
_HEALTH_FAILED = (AiAuthError, AiTimeout, AiProviderError)


class AnyLLMDriver:
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

    def _extra_name(self) -> str:
        """The arvel-ai extra that fixes a missing engine. Extras mirror any-llm's provider
        names, so the provider prefix of the configured model IS the extra to install
        (`anthropic:claude-...` -> `arvel-ai[anthropic]`); bare SDK when no model names one."""
        if self.model and ":" in self.model:
            return self.model.split(":", 1)[0]
        return "any-llm"

    def _any_llm(self) -> Any:
        try:
            # any-llm is an optional extra, deliberately not installed in dev (see module
            # docstring) — pyright can't see a package that isn't in the environment, and no
            # annotation fixes that; the import itself is erased to Any right below anyway.
            import any_llm  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise MissingExtraError(self._extra_name(), package="arvel-ai") from exc
        return any_llm

    async def health(self) -> HealthResult:
        """Check the provider by actually asking for a one-token completion, so it goes through
        the same auth and provider path a real call would. A bad or missing key -> failed, an
        unreachable or slow provider -> failed, a request-level error -> degraded, and a real
        completion -> ok. Non-critical, so a failure degrades startup rather than aborting it."""
        if not self.model:
            return HealthResult(HealthStatus.DEGRADED, detail="no model configured")
        try:
            any_llm = self._any_llm()
        except MissingExtraError as exc:
            return HealthResult(HealthStatus.FAILED, detail=str(exc))
        try:
            await asyncio.wait_for(
                any_llm.acompletion(
                    model=self.model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                ),
                _HEALTH_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001 - translated at the boundary
            err = self._translate(exc)
            status = (
                HealthStatus.FAILED if isinstance(err, _HEALTH_FAILED) else HealthStatus.DEGRADED
            )
            return HealthResult(status, detail=f"{type(err).__name__}: {err}")
        return HealthResult(HealthStatus.OK, detail="completion reachable")

    async def _call_with_retries(self, invoke: Callable[[], Awaitable[Any]]) -> Any:
        """One provider call under the driver timeout, retrying retryable failures with
        exponential backoff — any-llm has no retry or timeout knobs of its own."""
        for attempt in range(self.max_retries + 1):
            try:
                return await asyncio.wait_for(invoke(), self.timeout)
            except Exception as exc:  # noqa: BLE001 - translated at the boundary
                err = self._translate(exc)
                if not err.retryable or attempt == self.max_retries:
                    raise err from exc
                await asyncio.sleep(getattr(err, "retry_after", None) or 0.5 * 2**attempt)
        raise AssertionError("unreachable")  # pragma: no cover

    async def chat(self, request: ChatRequest) -> ChatResponse:
        any_llm = self._any_llm()
        payload = to_openai_payload(request, self.model)
        result = await self._call_with_retries(lambda: any_llm.acompletion(**payload))
        return parse_openai_response(
            result.model_dump() if hasattr(result, "model_dump") else dict(result),
            include_raw=self.include_raw,
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatDelta]:
        any_llm = self._any_llm()
        payload = to_openai_payload(request, self.model)
        text_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        model = ""
        finish = "stop"
        try:
            stream = aiter(
                await asyncio.wait_for(any_llm.acompletion(**payload, stream=True), self.timeout)
            )
            while True:
                try:
                    # per-chunk idle timeout — the whole-stream duration is open-ended
                    chunk = await asyncio.wait_for(anext(stream), self.timeout)
                except StopAsyncIteration:
                    break
                data: dict[str, Any] = (
                    chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
                )
                model = data.get("model") or model
                choices: list[dict[str, Any]] = data.get("choices") or [{}]
                choice = choices[0]
                finish = choice.get("finish_reason") or finish
                delta: dict[str, Any] = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    text_parts.append(content)
                    yield TextDelta(text=content)
                raw_tool_calls: list[dict[str, Any]] = delta.get("tool_calls") or []
                for tc in raw_tool_calls:
                    index = tc.get("index", 0)
                    slot = tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    fn: dict[str, Any] = tc.get("function") or {}
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
                stop_reason=FINISH_REASONS.get(finish, "other"),
                model=model,
            )
        )

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        any_llm = self._any_llm()
        result = await self._call_with_retries(
            lambda: any_llm.aembedding(model=request.model or self.model, inputs=request.texts)
        )
        data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        usage: dict[str, Any] = data.get("usage") or {}
        return EmbedResponse(
            vectors=[item["embedding"] for item in data.get("data", [])],
            model=data.get("model", ""),
            usage=Usage(input_tokens=usage.get("prompt_tokens", 0)),
        )

    # -- turning any-llm exceptions into AiErrors --------------------------------

    @staticmethod
    def _translate(exc: Exception) -> AiError:
        if isinstance(exc, AiError):
            return exc
        if isinstance(exc, TimeoutError):  # asyncio.wait_for — the driver-enforced timeout
            return AiTimeout(f"request exceeded the driver timeout: {exc}")
        translated = _match(exc)
        if translated is not None:
            return translated
        # any-llm wraps SDK errors but keeps the original — classify on it before giving up
        original = getattr(exc, "original_exception", None)
        if isinstance(original, Exception):
            translated = _match(original)
            if translated is not None:
                return translated
        return AiProviderError(f"{type(exc).__name__}: {exc}")


def _match(exc: Exception) -> AiError | None:
    """Classify by exception class name + status code — covers any-llm's own hierarchy and
    the OpenAI-convention names the underlying provider SDKs raise, without importing either."""
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    message = f"{name}: {exc}"
    if name in ("AuthenticationError", "MissingApiKeyError", "PermissionDeniedError") or status in (
        401,
        403,
    ):
        return AiAuthError(message)
    if name == "RateLimitError" or status == 429:
        return AiRateLimited(message)
    if name in ("GatewayTimeoutError", "Timeout", "APITimeoutError"):
        return AiTimeout(message)
    if name in (
        "ContentFilterError",
        "ContentFilterFinishReasonError",
        "ContentPolicyViolationError",
    ):
        return AiContentFiltered(message)
    if name in (
        "InvalidRequestError",
        "ModelNotFoundError",
        "UnsupportedParameterError",
        "UnsupportedProviderError",
        "ContextLengthExceededError",
        "BadRequestError",
        "NotFoundError",
        "UnprocessableEntityError",
    ) or (status is not None and 400 <= int(status) < 500):
        return AiInvalidRequest(message)
    return None
