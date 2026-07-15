"""Gateway contract tests — pin down the public API shape.

Everything here runs against the fake driver. It's a real registered driver, so `AI.fake()`
and swapping it in via config use the exact same mechanism production does.
"""

from __future__ import annotations

import msgspec
import pytest

from arvel.kernel import Application

from arvel_ai import AI
from arvel_ai.contracts import (
    AiCapabilityError,
    AiError,
    AiProviderError,
    AiRateLimited,
    ChatRequest,
    ChatResponse,
    Message,
    StreamEnd,
    Text,
    TextDelta,
    ToolCall,
    ToolDef,
    Usage,
)
from arvel_ai.drivers.fake import FakeAiDriver


# ---- contracts -----------------------------------------------------------


def test_message_accepts_str_and_parts() -> None:
    assert Message(role="user", content="hi").content == "hi"
    m = Message(
        role="assistant", content=[Text(text="a"), ToolCall(id="1", name="t", arguments={})]
    )
    assert isinstance(m.content[0], Text)


def test_chat_response_conveniences() -> None:
    resp = ChatResponse(
        content=[
            Text(text="hello "),
            ToolCall(id="1", name="t", arguments={"a": 1}),
            Text(text="world"),
        ],
        stop_reason="tool_use",
        model="m",
        usage=Usage(input_tokens=1, output_tokens=2),
    )
    assert resp.text == "hello world"
    assert [c.name for c in resp.tool_calls] == ["t"]


def test_structured_decodes_into_schema() -> None:
    class Answer(msgspec.Struct):
        value: int

    resp = ChatResponse(content=[Text(text='{"value": 7}')], stop_reason="end_turn", model="m")
    assert resp.structured(Answer).value == 7


def test_error_taxonomy_retryable_flags() -> None:
    assert not AiError("x").retryable
    assert AiRateLimited("x", retry_after=1.5).retryable
    assert AiRateLimited("x").retry_after is None
    assert AiProviderError("x").retryable
    assert issubclass(AiCapabilityError, AiError)


# ---- fake driver ---------------------------------------------------------


async def test_fake_driver_scripts_replies_and_records() -> None:
    fake = FakeAiDriver(replies=["first", "second"])
    r1 = await fake.chat(ChatRequest(messages=[Message(role="user", content="a")]))
    r2 = await fake.chat(ChatRequest(messages=[Message(role="user", content="b")]))
    r3 = await fake.chat(ChatRequest(messages=[Message(role="user", content="c")]))
    assert (r1.text, r2.text, r3.text) == ("first", "second", "second")  # last reply sticks
    assert len(fake.requests) == 3
    fake.assert_chatted("b")


async def test_fake_driver_streams_deltas_then_end() -> None:
    fake = FakeAiDriver(replies=["ab"])
    events = [
        e async for e in fake.stream(ChatRequest(messages=[Message(role="user", content="x")]))
    ]
    assert all(isinstance(e, TextDelta) for e in events[:-1])
    assert isinstance(events[-1], StreamEnd)
    assert "".join(d.text for d in events[:-1]) == "ab"
    assert events[-1].response.text == "ab"


async def test_fake_driver_embeds() -> None:
    fake = FakeAiDriver()
    out = await fake.embed_texts(["a", "bb"])
    assert len(out.vectors) == 2


# ---- manager + facade over a booted app -----------------------------------


@pytest.fixture()
def fake_app(app: Application) -> Application:
    """The booted app from conftest, pointed at the fake driver."""
    app.make("config").set("ai.default", "fake")
    return app


async def test_manager_chat_str_sugar_and_alias_resolution(fake_app: Application) -> None:
    manager = fake_app.make("ai")
    fake_app.make("config").set("ai.models.fast", "concrete-model-id")
    resp = await manager.chat("hello", model="fast")
    assert isinstance(resp, ChatResponse)
    sent = manager.driver().requests[-1]
    assert sent.model == "concrete-model-id"  # alias resolved before the driver
    assert sent.messages[0].content == "hello"  # str sugar became a user message


async def test_manager_structured(fake_app: Application) -> None:
    class Product(msgspec.Struct):
        name: str

    manager = fake_app.make("ai")
    manager.driver().replies = ['{"name": "socks"}']
    product = await manager.structured(Product, "make a product")
    assert product.name == "socks"
    assert manager.driver().requests[-1].response_schema is Product


async def test_manager_embed_capability_gate(fake_app: Application) -> None:
    manager = fake_app.make("ai")
    manager.driver().supports_embeddings = False
    with pytest.raises(AiCapabilityError):
        await manager.embed(["x"])


async def test_facade_fake_is_a_driver_swap() -> None:
    fake = AI.fake()
    assert isinstance(fake, FakeAiDriver)
    resp = await AI.chat(ChatRequest(messages=[Message(role="user", content="ping")]))
    assert resp.text == "ok"
    fake.assert_chatted("ping")
    AI.clear_swapped()


def test_tooldef_rides_on_the_request() -> None:
    req = ChatRequest(
        messages=[Message(role="user", content="x")],
        tools=[ToolDef(name="get_weather", description="d", input_schema={"type": "object"})],
        tool_choice="get_weather",
    )
    assert req.tools[0].name == "get_weather"
