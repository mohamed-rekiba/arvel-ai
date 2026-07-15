"""Live provider smoke tests through the litellm SDK — a real provider, not a mock. No
Docker needed, since litellm is itself the client; these hit a real provider directly.

Gated on a real key; skipped otherwise so the default suite stays hermetic:
    AI_LIVE_MODEL=anthropic/claude-haiku-4-5 ANTHROPIC_API_KEY=... \
        uv run pytest tests/test_live_providers.py -q
"""

from __future__ import annotations

import os

import pytest

from arvel_ai.contracts import ChatRequest, Message, StreamEnd
from arvel_ai.drivers.litellm import LiteLLMDriver

MODEL = os.environ.get("AI_LIVE_MODEL")

pytestmark = pytest.mark.skipif(
    not MODEL, reason="AI_LIVE_MODEL not set (needs a real provider key)"
)


@pytest.fixture()
def driver() -> LiteLLMDriver:
    return LiteLLMDriver(model=MODEL, timeout=120.0)


async def test_chat_against_real_provider(driver: LiteLLMDriver) -> None:
    response = await driver.chat(
        ChatRequest(messages=[Message(role="user", content="Reply with just: pong")])
    )
    assert "pong" in response.text.lower()
    assert response.usage.output_tokens > 0


async def test_stream_against_real_provider(driver: LiteLLMDriver) -> None:
    events = [
        e
        async for e in driver.stream(
            ChatRequest(messages=[Message(role="user", content="Count: 1 2 3")])
        )
    ]
    assert isinstance(events[-1], StreamEnd)
    assert events[-1].response.text
