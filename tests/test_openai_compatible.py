"""openai_compatible driver: translation + error mapping, exercised through a
real httpx client against an in-process transport. (The full contract suite
against a REAL dockerized endpoint lives behind AI_INTEGRATION_BASE_URL —
see test_integration_endpoint.py; DR-0043.)
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import msgspec
import pytest

from arvel_ai.contracts import (
    AiAuthError,
    AiInvalidRequest,
    AiRateLimited,
    ChatRequest,
    Message,
    StreamEnd,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolDef,
    ToolResult,
)
from arvel_ai.drivers._openai_format import to_openai_payload
from arvel_ai.drivers.openai_compatible import OpenAICompatibleDriver

CHAT_OK = {
    "model": "test-model",
    "choices": [
        {
            "finish_reason": "tool_calls",
            "message": {
                "content": "checking",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
}


def driver_with(handler: Any) -> OpenAICompatibleDriver:
    return OpenAICompatibleDriver(
        base_url="http://gateway.test/v1",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )


# ---- translation (request building) ----------------------------------------


def test_payload_carries_system_tools_and_schema() -> None:
    class Out(msgspec.Struct):
        name: str

    request = ChatRequest(
        messages=[
            Message(role="user", content="hi"),
            Message(
                role="user",
                content=[ToolResult(tool_call_id="call_1", content="sunny")],
            ),
        ],
        system="be brief",
        tools=[ToolDef(name="get_weather", description="d", input_schema={"type": "object"})],
        tool_choice="get_weather",
        response_schema=Out,
        max_tokens=64,
        options={"temperature": 0.2},
    )
    payload = to_openai_payload(request, "default-model")
    assert payload["messages"][0] == {"role": "system", "content": "be brief"}
    assert payload["messages"][2]["role"] == "tool"
    assert payload["tools"][0]["function"]["name"] == "get_weather"
    assert payload["tool_choice"]["function"]["name"] == "get_weather"
    assert payload["response_format"]["json_schema"]["schema"]["type"] == "object"
    assert payload["temperature"] == 0.2  # options passthrough
    assert payload["model"] == "default-model"  # request named no model -> driver default


# ---- round trip -------------------------------------------------------------


async def test_chat_round_trip_parses_tool_calls_and_usage() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=CHAT_OK)

    driver = driver_with(handler)
    response = await driver.chat(ChatRequest(messages=[Message(role="user", content="weather?")]))
    assert seen["path"] == "/v1/chat/completions"
    assert response.stop_reason == "tool_use"
    assert response.tool_calls == [
        ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"})
    ]
    assert response.usage.input_tokens == 10
    assert response.text == "checking"


async def test_stream_yields_deltas_then_end() -> None:
    sse = (
        'data: {"model":"m","choices":[{"delta":{"content":"He"}}]}\n\n'
        'data: {"model":"m","choices":[{"delta":{"content":"y"}}]}\n\n'
        'data: {"model":"m","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}
        )

    driver = driver_with(handler)
    events = [
        e async for e in driver.stream(ChatRequest(messages=[Message(role="user", content="x")]))
    ]
    assert [d.text for d in events[:-1] if isinstance(d, TextDelta)] == ["He", "y"]
    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.response.text == "Hey"
    assert end.response.stop_reason == "end_turn"


async def test_embed_round_trip() -> None:
    from arvel_ai.contracts import EmbedRequest

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        return httpx.Response(
            200,
            json={"model": "e", "data": [{"embedding": [0.1, 0.2]}], "usage": {"prompt_tokens": 3}},
        )

    out = await driver_with(handler).embed(EmbedRequest(texts=["a"]))
    assert out.vectors == [[0.1, 0.2]]
    assert out.usage.input_tokens == 3


# ---- error mapping ----------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "exc"),
    [(401, AiAuthError), (403, AiAuthError), (404, AiInvalidRequest), (400, AiInvalidRequest)],
)
async def test_http_errors_map_to_taxonomy(status: int, exc: type[Exception]) -> None:
    driver = driver_with(lambda request: httpx.Response(status, text="nope"))
    with pytest.raises(exc):
        await driver.chat(ChatRequest(messages=[Message(role="user", content="x")]))


async def test_rate_limit_carries_retry_after() -> None:
    driver = driver_with(
        lambda request: httpx.Response(429, text="slow down", headers={"retry-after": "2.5"})
    )
    with pytest.raises(AiRateLimited) as exc:
        await driver.chat(ChatRequest(messages=[Message(role="user", content="x")]))
    assert exc.value.retry_after == 2.5
    assert exc.value.retryable


async def test_retry_after_http_date_does_not_escape_taxonomy() -> None:
    driver = driver_with(
        lambda request: httpx.Response(
            429, text="slow", headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}
        )
    )
    with pytest.raises(AiRateLimited) as exc:  # not a raw ValueError
        await driver.chat(ChatRequest(messages=[Message(role="user", content="x")]))
    assert exc.value.retry_after is None


async def test_stream_transport_error_maps_to_taxonomy() -> None:
    from arvel_ai.contracts import AiProviderError

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection reset")

    driver = driver_with(handler)
    with pytest.raises(AiProviderError):  # not a raw httpx.ConnectError leaking out
        async for _ in driver.stream(ChatRequest(messages=[Message(role="user", content="x")])):
            pass


async def test_stream_skips_malformed_sse_chunk() -> None:
    sse = (
        'data: {"model":"m","choices":[{"delta":{"content":"a"}}]}\n\n'
        "data: {not json\n\n"
        'data: {"model":"m","choices":[{"delta":{"content":"b"},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )
    driver = driver_with(
        lambda request: httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}
        )
    )
    events = [
        e async for e in driver.stream(ChatRequest(messages=[Message(role="user", content="x")]))
    ]
    assert isinstance(events[-1], StreamEnd)
    assert events[-1].response.text == "ab"  # the bad chunk was skipped, not fatal


async def test_stream_emits_tool_call_deltas_and_buffers_full_call() -> None:
    chunks = [
        {
            "model": "m",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ]
                    }
                }
            ],
        },
        {
            "model": "m",
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]}}
            ],
        },
        {
            "model": "m",
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"NYC"}'}}]}}
            ],
        },
        {"model": "m", "choices": [{"finish_reason": "tool_calls", "delta": {}}]},
    ]
    sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    driver = driver_with(
        lambda request: httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}
        )
    )
    events = [
        e async for e in driver.stream(ChatRequest(messages=[Message(role="user", content="x")]))
    ]
    deltas = [e for e in events if isinstance(e, ToolCallDelta)]
    assert deltas[0].id == "call_1" and deltas[0].name == "get_weather"
    assert "".join(d.arguments for d in deltas) == '{"city":"NYC"}'  # fragments reassemble
    end = events[-1]
    assert isinstance(end, StreamEnd)
    # the complete call is still buffered into StreamEnd for consumers that skip the deltas
    assert end.response.tool_calls == [
        ToolCall(id="call_1", name="get_weather", arguments={"city": "NYC"})
    ]


async def test_health_probes_models_and_reports_real_status() -> None:
    from arvel.contracts import HealthStatus

    # a reachable, authorized gateway -> OK (GET /models, the liveness route)
    def ok_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "m"}]})

    assert (await driver_with(ok_handler).health()).status is HealthStatus.OK

    # a wrong/missing key -> 401 -> FAILED (this is the case that used to report OK)
    auth = await driver_with(lambda r: httpx.Response(401, text="bad key")).health()
    assert auth.status is HealthStatus.FAILED
    assert "auth" in (auth.detail or "")

    # an unreachable gateway -> FAILED
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    assert (await driver_with(boom).health()).status is HealthStatus.FAILED


async def test_health_falls_back_to_chat_when_models_rejects_auth() -> None:
    from arvel.contracts import HealthStatus

    # mirrors Anthropic's OpenAI-compat endpoint: GET /models 401s even with a valid key (it
    # wants x-api-key), but /chat/completions works — health must verify on the real path.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(401, text="use x-api-key")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        )

    result = await driver_with(handler).health()
    assert result.status is HealthStatus.OK
    assert "chat" in (result.detail or "")


async def test_health_degraded_when_no_base_url() -> None:
    from arvel.contracts import HealthStatus

    driver = OpenAICompatibleDriver(base_url=None)
    result = await driver.health()
    assert result.status is HealthStatus.DEGRADED  # not a false OK
    assert "AI_GATEWAY_URL" in (result.detail or "")


async def test_driver_pools_one_client_and_aclose_drains_it() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=CHAT_OK)

    driver = driver_with(handler)
    req = ChatRequest(messages=[Message(role="user", content="x")])
    await driver.chat(req)
    await driver.chat(req)
    assert calls == 2  # both calls served through the one pooled client
    # the pool's keep-alive client is live until drained
    assert driver._http._shared  # a per-loop client was created and cached
    await driver.aclose()
    assert not driver._http._shared  # aclose drained + cleared the pool
