"""OpenAI-format translation, shared by the openai_compatible and litellm
drivers (LiteLLM normalizes every provider to this shape).

Pure functions over stdlib + msgspec — no engine imports here.
"""

from __future__ import annotations

import json
from typing import Any

import msgspec

from arvel_ai.contracts import (
    AiProviderError,
    ChatRequest,
    ChatResponse,
    Message,
    StopReason,
    Text,
    ToolCall,
    ToolResult,
    Usage,
)


def decode_tool_arguments(raw: str) -> dict[str, Any]:
    """Parse a tool-call ``arguments`` string, turning malformed JSON into an ``AiProviderError``
    so a bad provider response surfaces as one of our errors instead of a raw ``JSONDecodeError``."""
    try:
        return json.loads(raw or "{}")  # type: ignore[no-any-return]
    except json.JSONDecodeError as exc:
        raise AiProviderError(f"provider sent invalid tool-call arguments: {exc}") from exc


_FINISH_REASONS: dict[str, StopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


def to_openai_messages(request: ChatRequest) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if request.system:
        out.append({"role": "system", "content": request.system})
    for message in request.messages:
        if isinstance(message.content, str):
            out.append({"role": message.role, "content": message.content})
            continue
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for part in message.content:
            if isinstance(part, Text):
                text_parts.append(part.text)
            elif isinstance(part, ToolCall):
                tool_calls.append(
                    {
                        "id": part.id,
                        "type": "function",
                        "function": {"name": part.name, "arguments": json.dumps(part.arguments)},
                    }
                )
            elif isinstance(part, ToolResult):
                # tool results are their own messages in OpenAI format
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": part.tool_call_id,
                        "content": part.content,
                    }
                )
        entry: dict[str, Any] = {"role": message.role, "content": "".join(text_parts) or None}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        if entry["content"] is not None or tool_calls:
            out.append(entry)
    return out


def to_openai_payload(request: ChatRequest, default_model: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model or default_model,
        "messages": to_openai_messages(request),
    }
    if request.tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in request.tools
        ]
        if request.tool_choice in ("none", "required"):
            payload["tool_choice"] = request.tool_choice
        elif request.tool_choice != "auto":
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": request.tool_choice},
            }
    if request.response_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": _json_schema(request.response_schema)},
        }
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens
    if request.stop:
        payload["stop"] = request.stop
    payload.update(request.options)  # provider passthrough always wins
    return payload


def _json_schema(schema: Any) -> dict[str, Any]:
    if isinstance(schema, dict):
        return schema
    generated = msgspec.json.schema(schema)
    # inline the single $ref msgspec emits, since not every provider resolves $refs
    defs = generated.pop("$defs", {})
    ref = generated.get("$ref", "")
    if ref.startswith("#/$defs/"):
        return dict(defs[ref[len("#/$defs/") :]])
    return generated


def parse_openai_response(payload: dict[str, Any], include_raw: bool = False) -> ChatResponse:
    choice = payload["choices"][0]
    message = choice.get("message", {})
    content: list[Any] = []
    if message.get("content"):
        content.append(Text(text=message["content"]))
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        arguments = function.get("arguments") or "{}"
        content.append(
            ToolCall(
                id=call.get("id", ""),
                name=function.get("name", ""),
                arguments=decode_tool_arguments(arguments)
                if isinstance(arguments, str)
                else arguments,
            )
        )
    usage = payload.get("usage") or {}
    return ChatResponse(
        content=content,
        stop_reason=_FINISH_REASONS.get(choice.get("finish_reason") or "", "other"),
        model=payload.get("model", ""),
        usage=Usage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cache_read_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        ),
        raw=payload if include_raw else None,
    )


__all__ = [
    "Message",
    "parse_openai_response",
    "to_openai_messages",
    "to_openai_payload",
]
