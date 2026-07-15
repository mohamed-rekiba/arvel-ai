"""Driver for any OpenAI-format endpoint: a deployed LiteLLM proxy, vLLM,
Ollama's OpenAI route, or OpenAI itself.

HTTP + SSE go through arvel's ``Http`` client (``arvel.client.Client``): arvel core owns
connection pooling, timeouts, and SSE parsing (``ServerSentEvent``). The driver holds one
pooled ``Client`` and reuses its keep-alive connections across calls; ``aclose()`` drains
the pool at shutdown (DR-0039 — ``AiResource.disconnect`` calls it). This driver is the
anti-corruption boundary (DR-0041) — it maps arvel's transport exceptions
(``RequestTimedOut``/``TransportFailed``) and OpenAI status codes to the ``AiError``
taxonomy so no engine type crosses the public surface. It never imports httpx: arvel's
client fully wraps the engine (DR-0044).

The API key comes from the env var NAMED in config (api_key_env) — never from
config values.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from arvel.client import Client, PendingRequest, RequestFailed, RequestTimedOut, TransportFailed

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
    ToolCallDelta,
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
        transport: Any = None,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_key_env = api_key_env
        self.model = model
        self.timeout = timeout
        self.include_raw = include_raw
        # one pooled client for the driver's lifetime — keep-alive connections are reused
        # across calls; drained by aclose() at shutdown (DR-0039)
        self._http = Client(transport=transport)

    def _client(self) -> PendingRequest:
        if not self.base_url:
            raise AiInvalidRequest(
                "openai_compatible driver has no base_url - set config ai.drivers.openai_compatible.base_url"
            )
        req = self._http.base_url(self.base_url).timeout(self.timeout)
        key = os.environ.get(self.api_key_env, "")
        if key:
            req = req.with_token(key)
        return req

    async def aclose(self) -> None:
        """Drain the pooled client's keep-alive connections (DR-0039 teardown)."""
        await self._http.aclose()

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
                    index = tc.get("index", 0)
                    slot = tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    tc_id, name = tc.get("id"), fn.get("name")
                    args_fragment = fn.get("arguments") or ""
                    slot["id"] = tc_id or slot["id"]
                    slot["name"] = name or slot["name"]
                    slot["arguments"] += args_fragment
                    # stream the fragment; the full calls are still buffered into StreamEnd
                    yield ToolCallDelta(index=index, id=tc_id, name=name, arguments=args_fragment)
        except RequestFailed as exc:  # non-2xx status — map it like _post does
            resp = exc.response
            self._raise_for_status(resp.status(), resp.body(), resp.header("retry-after"))
        except RequestTimedOut as exc:
            raise AiTimeout(str(exc)) from exc
        except TransportFailed as exc:  # don't leak the engine type past the boundary (DR-0041)
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
        try:
            response = await self._client().post(path, json=payload)
        except RequestTimedOut as exc:
            raise AiTimeout(str(exc)) from exc
        except TransportFailed as exc:
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
