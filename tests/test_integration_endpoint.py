"""Contract suite against a REAL OpenAI-compatible endpoint (DR-0043).

Gated by AI_INTEGRATION_BASE_URL — start one with:
    docker compose -f docker-compose.test.yml up -d
    AI_INTEGRATION_BASE_URL=http://localhost:4000 uv run pytest tests/test_integration_endpoint.py -q
"""

from __future__ import annotations

import os

import pytest

from arvel_ai.contracts import ChatRequest, EmbedRequest, Message, StreamEnd
from arvel_ai.drivers.openai_compatible import OpenAICompatibleDriver

BASE_URL = os.environ.get("AI_INTEGRATION_BASE_URL")
MODEL = os.environ.get("AI_INTEGRATION_MODEL", "integration-test")

pytestmark = pytest.mark.skipif(
    not BASE_URL, reason="AI_INTEGRATION_BASE_URL not set (start docker-compose.test.yml)"
)


@pytest.fixture()
def driver() -> OpenAICompatibleDriver:
    return OpenAICompatibleDriver(
        base_url=f"{BASE_URL}/v1" if not str(BASE_URL).endswith("/v1") else str(BASE_URL),
        api_key_env="AI_INTEGRATION_API_KEY",
        model=MODEL,
        timeout=120.0,
    )


async def test_chat_against_real_endpoint(driver: OpenAICompatibleDriver) -> None:
    response = await driver.chat(
        ChatRequest(messages=[Message(role="user", content="Reply with the single word: pong")])
    )
    assert "pong" in response.text.lower()
    assert response.usage.output_tokens > 0


async def test_stream_against_real_endpoint(driver: OpenAICompatibleDriver) -> None:
    events = [
        e
        async for e in driver.stream(
            ChatRequest(messages=[Message(role="user", content="Count: 1 2 3")])
        )
    ]
    assert isinstance(events[-1], StreamEnd)
    assert events[-1].response.text


async def test_embed_against_real_endpoint(driver: OpenAICompatibleDriver) -> None:
    if os.environ.get("AI_INTEGRATION_EMBED_MODEL") is None:
        pytest.skip("no embedding model configured on the endpoint")
    out = await driver.embed(
        EmbedRequest(texts=["hello"], model=os.environ["AI_INTEGRATION_EMBED_MODEL"])
    )
    assert out.vectors and len(out.vectors[0]) > 8
