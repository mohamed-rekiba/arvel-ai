"""any-llm driver — exercised with a fake `any_llm` module. The real SDK is an optional
extra that isn't installed in dev; the live provider path runs in the provider tier."""

from __future__ import annotations

import asyncio
from typing import Any

from arvel.contracts import HealthStatus
import pytest

from arvel_ai.contracts import (
    AiAuthError,
    AiInvalidRequest,
    AiTimeout,
    ChatRequest,
    EmbedRequest,
    Message,
    StreamEnd,
    TextDelta,
    ToolCallDelta,
)
from arvel_ai.drivers.any_llm import AnyLLMDriver


class AuthenticationError(Exception):
    """Name matches any-llm's exception — AnyLLMDriver._translate keys off the class name."""


class RateLimitError(Exception):
    """Retryable by name; the driver's retry loop should absorb one of these."""


class InvalidRequestError(Exception):
    """Non-retryable by name; the driver must raise immediately, no retry."""


class ProviderError(Exception):
    """any-llm's generic wrapper — carries the real SDK error on `original_exception`."""

    def __init__(self, message: str, original_exception: Exception) -> None:
        super().__init__(message)
        self.original_exception = original_exception


class _FakeAnyLLM:
    """Stands in for the any_llm module: an async `acompletion` that returns a canned result,
    raises queued exceptions (each once, in order — so retry paths can recover), or
    (stream=True) yields canned chunks; `aembedding` returns a canned embeddings payload."""

    def __init__(
        self,
        *,
        result: Any = None,
        raises: list[Exception] | None = None,
        chunks: list[dict[str, Any]] | None = None,
        embeddings: Any = None,
    ) -> None:
        self._result = result
        self._raises = list(raises or [])
        self._chunks = chunks or []
        self._embeddings = embeddings
        self.completion_calls = 0

    async def acompletion(self, **kwargs: Any) -> Any:
        self.completion_calls += 1
        if self._raises:
            raise self._raises.pop(0)
        if kwargs.get("stream"):

            async def _gen() -> Any:
                for chunk in self._chunks:
                    yield chunk

            return _gen()
        return self._result

    async def aembedding(self, **kwargs: Any) -> Any:
        if self._raises:
            raise self._raises.pop(0)
        return self._embeddings


def _driver_with(
    fake: Any, model: str | None = "anthropic:claude-x", max_retries: int = 2
) -> AnyLLMDriver:
    driver = AnyLLMDriver(model=model, max_retries=max_retries)
    driver._any_llm = lambda: fake  # type: ignore[method-assign]
    return driver


async def test_health_ok_on_successful_completion() -> None:
    driver = _driver_with(_FakeAnyLLM(result={"choices": [{"message": {"content": "ok"}}]}))
    assert (await driver.health()).status is HealthStatus.OK


async def test_health_failed_on_bad_key() -> None:
    driver = _driver_with(_FakeAnyLLM(raises=[AuthenticationError("invalid key")]))
    result = await driver.health()
    assert result.status is HealthStatus.FAILED  # not a false OK
    assert "Auth" in (result.detail or "")


async def test_health_degraded_without_a_model() -> None:
    assert (await _driver_with(_FakeAnyLLM(), model=None).health()).status is HealthStatus.DEGRADED


async def test_stream_yields_text_deltas_then_end() -> None:
    chunks = [
        {"model": "m", "choices": [{"delta": {"content": "Hel"}}]},
        {"model": "m", "choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
    ]
    driver = _driver_with(_FakeAnyLLM(chunks=chunks))
    events = [
        e async for e in driver.stream(ChatRequest(messages=[Message(role="user", content="x")]))
    ]
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Hel", "lo"]
    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.response.text == "Hello"


async def test_stream_emits_tool_call_deltas_and_buffers_the_call() -> None:
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "lookup", "arguments": '{"q":'},
                            }
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"x"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    driver = _driver_with(_FakeAnyLLM(chunks=chunks))
    events = [
        e async for e in driver.stream(ChatRequest(messages=[Message(role="user", content="x")]))
    ]
    deltas = [e for e in events if isinstance(e, ToolCallDelta)]
    assert deltas[0].name == "lookup"
    assert "".join(d.arguments for d in deltas) == '{"q":"x"}'
    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.response.tool_calls[0].arguments == {"q": "x"}


async def test_chat_retries_a_rate_limit_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _instant(_delay: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _instant)
    fake = _FakeAnyLLM(
        result={"choices": [{"message": {"content": "ok"}}]},
        raises=[RateLimitError("slow down")],
    )
    driver = _driver_with(fake, max_retries=1)
    response = await driver.chat(ChatRequest(messages=[Message(role="user", content="x")]))
    assert response.text == "ok"
    assert fake.completion_calls == 2  # the failure plus the successful retry


async def test_chat_does_not_retry_non_retryable_errors() -> None:
    fake = _FakeAnyLLM(raises=[InvalidRequestError("bad model"), InvalidRequestError("again")])
    driver = _driver_with(fake, max_retries=2)
    with pytest.raises(AiInvalidRequest):
        await driver.chat(ChatRequest(messages=[Message(role="user", content="x")]))
    assert fake.completion_calls == 1


def test_translate_unwraps_the_original_exception() -> None:
    wrapped = ProviderError("provider blew up", AuthenticationError("bad key"))
    assert isinstance(AnyLLMDriver._translate(wrapped), AiAuthError)


async def test_stream_idle_timeout_becomes_ai_timeout() -> None:
    class _Hanging:
        async def acompletion(self, **kwargs: Any) -> Any:
            async def _gen() -> Any:
                await asyncio.sleep(5)
                yield {}

            return _gen()

    driver = AnyLLMDriver(model="anthropic:claude-x", timeout=0.05)
    driver._any_llm = lambda: _Hanging()  # type: ignore[method-assign]
    with pytest.raises(AiTimeout):
        _ = [
            e
            async for e in driver.stream(ChatRequest(messages=[Message(role="user", content="x")]))
        ]


async def test_embed_parses_vectors_and_usage() -> None:
    fake = _FakeAnyLLM(
        embeddings={
            "data": [{"embedding": [0.1, 0.2]}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 3},
        }
    )
    driver = _driver_with(fake)
    response = await driver.embed(EmbedRequest(texts=["hi"]))
    assert response.vectors == [[0.1, 0.2]]
    assert response.model == "text-embedding-3-small"
    assert response.usage.input_tokens == 3
