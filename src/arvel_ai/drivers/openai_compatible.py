"""Driver for any OpenAI-format endpoint: a deployed LiteLLM proxy, vLLM,
Ollama's OpenAI route, or OpenAI itself.

HTTP + SSE go through arvel's ``Http`` client (``arvel.client.PendingRequest``): arvel
core owns pooling, timeouts, and SSE parsing (``ServerSentEvent``). This driver is the
anti-corruption boundary (DR-0041) — it maps transport errors and OpenAI status codes to
the ``AiError`` taxonomy so no engine type crosses the public surface. It still catches
httpx exception *types* for transport failures (timeouts/connection resets), which arvel
surfaces raw; httpx is arvel core and the import-linter permits it in exactly this module.

The API key comes from the env var NAMED in config (api_key_env) — never from
config values.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from arvel.client import PendingRequest, RequestFailed

from arvel_ai.contracts import (
    AiAuthError,
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
    Usage,
)

from ._openai_format import parse_openai_response, to_openai_payload


def _parse_retry_after(value: str | None) -> float | None:
    # Retry-After may be seconds or an HTTP-date; we only surface the numeric form
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


class OpenAICompatibleDriver:
    supports_embeddings = True

    def __init__(
        self,
        base_url: str | None = None,
        api_key_env: str = "AI_API_KEY",
        model: str | None = None,
        timeout: float = 60.0,
        include_raw: bool = False,
        transport: Any = None,  # test seam: an httpx.MockTransport
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_key_env = api_key_env
        self.model = model
        self.timeout = timeout
        self.include_raw = include_raw
        self._transport = transport

    def _client(self) -> PendingRequest:
        if not self.base_url:
            raise AiInvalidRequest(
                "openai_compatible driver has no base_url - set config ai.drivers.openai_compatible.base_url"
            )
        req = (
            PendingRequest(transport=self._transport).base_url(self.base_url).timeout(self.timeout)
        )
        key = os.environ.get(self.api_key_env, "")
        if key:
            req = req.with_token(key)
        return req

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = to_openai_payload(request, self.model)
        data = await self._post("/chat/completions", payload)
        return parse_openai_response(data, include_raw=self.include_raw)

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatDelta]:
        payload = to_openai_payload(request, self.model) | {"stream": True}
        text_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish = "stop"
        model = ""
        import httpx

        try:
            async for event in self._client().stream("POST", "/chat/completions", json=payload):
                if event.data == "[DONE]":
                    break
                try:
                    chunk = json.loads(event.data)
                except json.JSONDecodeError:
                    continue  # skip a partial/keepalive chunk rather than crash the stream
                model = chunk.get("model", model)
                choice = (chunk.get("choices") or [{}])[0]
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    text_parts.append(delta["content"])
                    yield TextDelta(text=delta["content"])
                for tc in delta.get("tool_calls") or []:
                    slot = tool_calls.setdefault(
                        tc.get("index", 0), {"id": "", "name": "", "arguments": ""}
                    )
                    slot["id"] = tc.get("id") or slot["id"]
                    fn = tc.get("function") or {}
                    slot["name"] = fn.get("name") or slot["name"]
                    slot["arguments"] += fn.get("arguments") or ""
        except RequestFailed as exc:  # non-2xx status — map it like _post does
            resp = exc.response
            self._raise_for_status(resp.status(), resp.body(), resp.header("retry-after"))
        except httpx.TimeoutException as exc:
            raise AiTimeout(str(exc)) from exc
        except httpx.HTTPError as exc:
            # a transport failure must not leak the engine type across the public
            # surface (DR-0041) — map to the taxonomy like _post does
            raise AiProviderError(str(exc)) from exc
        content: list[Any] = [Text(text="".join(text_parts))] if text_parts else []
        for slot in tool_calls.values():
            content.append(
                ToolCall(
                    id=slot["id"],
                    name=slot["name"],
                    arguments=json.loads(slot["arguments"] or "{}"),
                )
            )
        from ._openai_format import _FINISH_REASONS  # noqa: PLC0415

        yield StreamEnd(
            response=ChatResponse(
                content=content,
                stop_reason=_FINISH_REASONS.get(finish, "other"),
                model=model,
            )
        )

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        payload = {"input": request.texts, "model": request.model or self.model}
        data = await self._post("/embeddings", payload)
        usage = data.get("usage") or {}
        return EmbedResponse(
            vectors=[item["embedding"] for item in data.get("data", [])],
            model=data.get("model", ""),
            usage=Usage(input_tokens=usage.get("prompt_tokens", 0)),
        )

    # -- error mapping (S1 taxonomy) ------------------------------------------

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx

        try:
            response = await self._client().post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise AiTimeout(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise AiProviderError(str(exc)) from exc
        if response.failed():
            self._raise_for_status(
                response.status(), response.body(), response.header("retry-after")
            )
        return dict(response.json())

    @staticmethod
    def _raise_for_status(status: int, body: str, retry_after: str | None = None) -> None:
        detail = f"HTTP {status}: {body[:300]}"
        if status in (401, 403):
            raise AiAuthError(detail)
        if status == 429:
            raise AiRateLimited(detail, retry_after=_parse_retry_after(retry_after))
        if status >= 500:
            raise AiProviderError(detail)
        raise AiInvalidRequest(detail)
