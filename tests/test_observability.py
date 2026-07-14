"""Lifecycle events reach the host app's dispatcher; spans no-op when tracing
is off (the arvel.telemetry gate — exporter-level observation happens in the
consumer-path evidence, where the telemetry extra is installed)."""

from __future__ import annotations

import pytest

from arvel.events.dispatcher import Dispatcher
from arvel.kernel import Application

from arvel_ai.events import AiRequestSending, AiResponseReceived
from arvel_ai.provider import AiServiceProvider


@pytest.fixture()
def eventful_app() -> Application:
    application = Application()
    application.singleton("events", lambda c: Dispatcher())
    provider = AiServiceProvider(application)
    provider.register()
    provider.boot()
    application.make("config").get("ai")["default"] = "fake"
    return application


async def test_chat_dispatches_lifecycle_events(eventful_app: Application) -> None:
    captured: list[object] = []
    events = eventful_app.make("events")
    events.listen(AiRequestSending, captured.append)
    events.listen(AiResponseReceived, captured.append)

    manager = eventful_app.make("ai")
    await manager.chat("hello")

    kinds = [type(e).__name__ for e in captured]
    assert kinds == ["AiRequestSending", "AiResponseReceived"]
    received = captured[1]
    assert isinstance(received, AiResponseReceived)
    assert received.driver == "fake"
    assert received.response.text == "ok"


async def test_stream_dispatches_response_at_stream_end(eventful_app: Application) -> None:
    captured: list[object] = []
    eventful_app.make("events").listen(AiResponseReceived, captured.append)

    manager = eventful_app.make("ai")
    async for _ in manager.stream("hi"):
        pass
    assert len(captured) == 1


async def test_chat_works_without_events_binding(app: Application) -> None:
    """A bare app (no events dispatcher) must not break the gateway."""
    app.make("config").get("ai")["default"] = "fake"
    response = await app.make("ai").chat("hello")
    assert response.text == "ok"