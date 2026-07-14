"""Driver for any OpenAI-format endpoint: a deployed LiteLLM proxy, vLLM,
Ollama's OpenAI route, or OpenAI itself. Engine: httpx (uv add 'arvel-ai[httpx]').

The API key comes from the env var NAMED in config (api_key_env) — never from
config values.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from arvel.support.manager import MissingExtraError

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

    # ponytail: a client per call; pool via the DR-0039 lifecycle when it matters
    def _httpx(self) -> Any:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise MissingExtraError("openai_compatible", extra="httpx", package="arvel-ai") from exc
        if not self.base_url:
            raise AiInvalidRequest(
                "openai_compatible driver has no base_url - set config ai.drivers.openai_compatible.base_url"
            )
        headers = {}
        key = os.environ.get(self.api_key_env, "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
            transport=self._transport,
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = to_openai_payload(request, self.model)
        async with self._httpx() as client:
            response = await self._post(client, "/chat/completions", payload)
        return parse_openai_response(response, include_raw=self.include_raw)

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatDelta]:
        payload = to_openai_payload(request, self.model) | {"stream": True}
        text_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish = "stop"
        model = ""
        async with self._httpx() as client:
            try:
                async with client.stream("POST", "/chat/completions", json=payload) as resp:
                    if resp.status_code >= 400:
                        await resp.aread()
                        self._raise_for_status(resp.status_code, resp.text)
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
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
            except TimeoutError as exc:
                raise AiTimeout(str(exc)) from exc
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
        async with self._httpx() as client:
            data = await self._post(client, "/embeddings", payload)
        usage = data.get("usage") or {}
        return EmbedResponse(
            vectors=[item["embedding"] for item in data.get("data", [])],
            model=data.get("model", ""),
            usage=Usage(input_tokens=usage.get("prompt_tokens", 0)),
        )

    # -- error mapping (S1 taxonomy) ------------------------------------------

    async def _post(self, client: Any, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx

        try:
            response = await client.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise AiTimeout(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise AiProviderError(str(exc)) from exc
        if response.status_code >= 400:
            retry_after = response.headers.get("retry-after")
            self._raise_for_status(response.status_code, response.text, retry_after)
        return dict(response.json())

    @staticmethod
    def _raise_for_status(
        status: int, body: str, retry_after: str | None = None
    ) -> None:
        detail = f"HTTP {status}: {body[:300]}"
        if status in (401, 403):
            raise AiAuthError(detail)
        if status == 429:
            raise AiRateLimited(detail, retry_after=float(retry_after) if retry_after else None)
        if status >= 500:
            raise AiProviderError(detail)
        raise AiInvalidRequest(detail)
