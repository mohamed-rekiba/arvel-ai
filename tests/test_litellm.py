"""LiteLLM driver — exercised with a fake `litellm` module. The real SDK is an optional
extra that isn't installed in dev; the live provider path runs in the provider tier."""

from __future__ import annotations

from typing import Any

from arvel.contracts import HealthStatus

from arvel_ai.contracts import ChatRequest, Message, StreamEnd, TextDelta, ToolCallDelta
from arvel_ai.drivers.litellm import LiteLLMDriver


class AuthenticationError(Exception):
    """Name matches litellm's exception — LiteLLMDriver._translate keys off the class name."""


class _FakeLiteLLM:
    """Stands in for the litellm module: an async `acompletion` that returns a canned result,
    raises, or (stream=True) yields canned chunks."""

    def __init__(
        self,
        *,
        result: Any = None,
        raises: Exception | None = None,
        chunks: list[dict[str, Any]] | None = None,
    ) -> None:
        self._result = result
        self._raises = raises
        self._chunks = chunks or []

    async def acompletion(self, **kwargs: Any) -> Any:
        if self._raises is not None:
            raise self._raises
        if kwargs.get("stream"):

            async def _gen() -> Any:
                for chunk in self._chunks:
                    yield chunk

            return _gen()
        return self._result


def _driver_with(fake: _FakeLiteLLM, model: str | None = "claude-x") -> LiteLLMDriver:
    driver = LiteLLMDriver(model=model)
    driver._litellm = lambda: fake  # type: ignore[method-assign]
    return driver


async def test_health_ok_on_successful_completion() -> None:
    driver = _driver_with(_FakeLiteLLM(result={"choices": [{"message": {"content": "ok"}}]}))
    assert (await driver.health()).status is HealthStatus.OK


async def test_health_failed_on_bad_key() -> None:
    driver = _driver_with(_FakeLiteLLM(raises=AuthenticationError("invalid key")))
    result = await driver.health()
    assert result.status is HealthStatus.FAILED  # not a false OK
    assert "Auth" in (result.detail or "")


async def test_health_degraded_without_a_model() -> None:
    assert (await _driver_with(_FakeLiteLLM(), model=None).health()).status is HealthStatus.DEGRADED


async def test_stream_yields_text_deltas_then_end() -> None:
    chunks = [
        {"model": "m", "choices": [{"delta": {"content": "Hel"}}]},
        {"model": "m", "choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
    ]
    driver = _driver_with(_FakeLiteLLM(chunks=chunks))
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
                            {"index": 0, "id": "c1", "function": {"name": "lookup", "arguments": '{"q":'}}
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"x"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    driver = _driver_with(_FakeLiteLLM(chunks=chunks))
    events = [
        e async for e in driver.stream(ChatRequest(messages=[Message(role="user", content="x")]))
    ]
    deltas = [e for e in events if isinstance(e, ToolCallDelta)]
    assert deltas[0].name == "lookup"
    assert "".join(d.arguments for d in deltas) == '{"q":"x"}'
    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.response.tool_calls[0].arguments == {"q": "x"}
