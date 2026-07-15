# The Gateway

One stable API over many AI providers. The gateway is the `AI` facade (backed by `AiManager`); it
gives your app chat, streaming, structured output, tool definitions, and embeddings through the
house driver pattern — swap providers in config, never in code. No provider SDK type appears
anywhere in these shapes.

```python
from arvel_ai import AI
```

All four methods are async.

## Chat

A plain string becomes a single user message:

```python
reply = await AI.chat("Summarize this review: ...", model="fast")
reply.text          # the assistant's text
reply.stop_reason   # end_turn | max_tokens | tool_use | refusal | other
reply.usage         # Usage(input_tokens, output_tokens, cache_read_tokens)
reply.model         # the concrete model id the provider reported
```

Full control uses the same arvel-owned shapes everywhere:

```python
from arvel_ai import ChatRequest, Message

reply = await AI.chat(
    [Message(role="user", content="Three names for a sock brand")],
    system="You are terse.",
    max_tokens=200,
    stop=["\n\n"],
    options={"temperature": 0.8},   # provider passthrough — not part of the stable surface
)
```

The first argument accepts three shapes, so you can grow from a string to a full request without
changing anything else:

```python
await AI.chat("just a string")                       # one user message
await AI.chat([Message(role="user", content="hi")])  # an explicit message list
await AI.chat(ChatRequest(messages=[...], system="…"))
```

Any keyword after `messages` is a `ChatRequest` field: `model`, `system`, `tools`, `tool_choice`,
`response_schema`, `max_tokens`, `stop`, `options`.

## Streaming

`AI.stream` returns an async iterator — iterate it, don't `await` it. It yields deltas as they
arrive and ends with a `StreamEnd` carrying the fully-assembled response:

```python
from arvel_ai import TextDelta, StreamEnd
from arvel_ai.contracts import ToolCallDelta

async for delta in AI.stream("Write a product description"):
    match delta:
        case TextDelta(text=chunk):
            print(chunk, end="", flush=True)
        case ToolCallDelta(index=i, arguments=fragment):
            ...                       # a tool call's arguments, streamed in fragments — join per index
        case StreamEnd(response=response):
            record(response.usage)    # a complete ChatResponse; tool_calls are fully assembled here
```

The complete tool calls are always buffered into `StreamEnd.response.tool_calls`, so a consumer that
doesn't need token-level tool streaming can ignore `ToolCallDelta` and read the end.

## Structured output

Define the shape as a msgspec `Struct`; the gateway asks the provider for schema-constrained output
and decodes it into a typed instance — not a dict, not `Any`:

```python
import msgspec

class ProductCopy(msgspec.Struct):
    title: str
    bullets: list[str]

copy = await AI.structured(ProductCopy, "Write copy for red wool socks", model="smart")
copy.title          # str — statically typed as ProductCopy, validated at runtime
```

## Tools

Tool definitions and calls ride on the request/response — your agent loop, your control:

```python
from arvel_ai import ToolDef, ToolResult, Message

reply = await AI.chat(
    "What's the weather in Paris?",
    tools=[ToolDef(name="get_weather", description="Current weather for a city",
                   input_schema={"type": "object", "properties": {"city": {"type": "string"}}})],
    tool_choice="auto",              # auto | none | required | "<tool name>"
)

# the model asked to call a tool — dispatch it yourself, then feed the result back
async def dispatch(name: str, arguments: dict) -> str:
    if name == "get_weather":
        return await weather.current(arguments["city"])
    raise ValueError(f"unknown tool {name!r}")

messages = [Message(role="user", content="What's the weather in Paris?")]
for call in reply.tool_calls:        # list[ToolCall]
    result = await dispatch(call.name, call.arguments)
    messages.append(Message(role="user", content=[
        ToolResult(tool_call_id=call.id, content=result)]))
final = await AI.chat(messages, tools=[...])
```

Keep tool `input_schema` to the JSON-Schema basics every provider accepts.

## Embeddings

```python
result = await AI.embed(["first text", "second text"], model="text-embedding-3-small")
result.vectors          # list[list[float]]
result.usage.input_tokens
```

Embeddings are per-driver: a driver whose provider has no embeddings endpoint raises
`AiCapabilityError` — route embeddings through a driver that supports them.

## Drivers & model aliases

`config("ai.default")` picks the driver; `config("ai.drivers.<name>")` configures it. The engine
(`httpx`, `litellm`) is lazy-imported *inside* the driver and never reaches the public surface.

```python
# config/ai.py
ai = {
    "default": "openai_compatible",
    "models": {"fast": "claude-haiku-4-5", "smart": "claude-opus-4-8"},
    "drivers": {
        "openai_compatible": {"base_url": "https://api.anthropic.com/v1", "api_key_env": "AI_API_KEY"},
    },
}
```

- **`openai_compatible`** — any OpenAI-format endpoint: a provider directly (Anthropic, OpenAI), a
  LiteLLM proxy, vLLM, or Ollama. Ships in the base install (its httpx engine is arvel core) and
  runs on arvel's own pooled HTTP client. Needs `base_url`; key via the env var named in
  `api_key_env`.
- **`litellm`** — the LiteLLM SDK: 100+ providers, keys via each provider's own env var
  (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …). Needs `uv add 'arvel-ai[litellm]'`.
- **`fake`** — the test double (see [Testing](#testing)).

`models` is the churn shield: apps say `model="fast"`; a provider retiring a model is one config
edit, zero code changes. An unmapped `model` passes through to the provider unchanged, so you can
always name a concrete id directly.

### Writing your own driver

Implement the `AiDriver` protocol — three methods — and register it on the manager; nothing else in
the package changes:

```python
from arvel_ai.contracts import AiDriver, ChatRequest, ChatResponse, EmbedRequest, EmbedResponse

class MyDriver:                       # structurally an AiDriver
    supports_embeddings = False

    async def chat(self, request: ChatRequest) -> ChatResponse: ...
    def stream(self, request: ChatRequest): ...        # -> AsyncIterator[ChatDelta]
    async def embed(self, request: EmbedRequest) -> EmbedResponse: ...
```

Translate your provider's errors into the [`AiError` taxonomy](#errors) at the boundary, and keep
the engine import *inside* the methods (add it as an extra in `pyproject.toml` so the import-linter
stays green). That's the whole contract.

## Health checks

The gateway registers as an arvel *resource*, so it appears in the startup log and on `/health`.
The check is **real** — it reaches the provider, it doesn't just guess from config:

- `openai_compatible` probes `GET /models`, falling back to a one-token chat on `/chat/completions`
  when that route rejects the auth (Anthropic's does, even with a valid key).
- `litellm` runs a one-token completion through the SDK.

`ok` means reachable *and* the key works; `failed` means a wrong/missing key or an unreachable
provider; `degraded` means unconfigured (no `base_url`/model). The resource is **non-critical** by
default (`config("ai.critical") = False`), so an AI outage degrades startup rather than aborting it —
set `critical: True` if your app can't run without AI.

## Errors

One taxonomy, whatever the provider — drivers translate their engine's exceptions at the boundary,
so a raw `httpx`/`litellm` error never reaches you. Each inherits `AiError` and carries a
`.retryable` flag:

| Error | When | retryable |
|-------|------|-----------|
| `AiAuthError` | bad/missing key (401/403) | no |
| `AiInvalidRequest` | malformed request (4xx) | no |
| `AiRateLimited` | throttled (429); carries `.retry_after` | yes |
| `AiProviderError` | provider 5xx / unreachable / bad body | yes |
| `AiTimeout` | request timed out | yes |
| `AiContentFiltered` | pre-output refusal / content filter | no |
| `AiCapabilityError` | the driver can't do this (e.g. embeddings) | no |

```python
import asyncio
from arvel_ai import AiRateLimited, AiError

try:
    reply = await AI.chat("...")
except AiRateLimited as exc:
    await asyncio.sleep(exc.retry_after or 1)   # back off, then retry
except AiError as exc:
    ...                                          # exc.retryable tells you whether to
```

## Observability

Every call opens a gated telemetry span (`ai.chat` / `ai.stream` / `ai.embed`, with token usage as
attributes — a no-op unless the `telemetry` extra is configured) and dispatches events your app can
hook:

```python
from arvel import Event
from arvel_ai.events import AiResponseReceived

Event.listen(AiResponseReceived, audit_ai_usage)
```

## Testing

The fake is a first-class driver, so tests swap it in exactly the way production swaps providers.
`AI.fake()` returns the `FakeAiDriver`; `AI.clear_swapped()` restores the real one (do it in a
fixture).

```python
from arvel_ai import AI

async def test_review_summary():
    fake = AI.fake()
    fake.replies = ["Great quality, runs small."]      # each call pops the next; the last one sticks
    reply = await AI.chat("summarize this review")
    assert reply.text == "Great quality, runs small."
    fake.assert_chatted("summarize")                   # some request's messages contained this
    AI.clear_swapped()
```

`fake.requests` holds every `ChatRequest` sent (inspect `.model`, `.messages`, `.tools`);
`fake.embedded` holds every `AI.embed` batch. For structured output, script a JSON string:
`fake.replies = ['{"title": "…", "bullets": []}']`. Full patterns — including a clean-up fixture —
are in the sections above and in each feature's own examples.

## Common mistakes & gotchas

- **Set `default` explicitly.** The package's built-in default driver is `litellm` (which needs
  `uv add 'arvel-ai[litellm]'`). A base `uv add arvel-ai` with no config hits a missing-extra error
  on the first call — set `"default": "openai_compatible"` to use the driver that ships with the
  base install.
- **Keys live in env vars.** Config holds the env var *name* (`api_key_env`), never the value.
- **`options` is passthrough, not stable API.** Some providers reject sampling params entirely;
  anything in `options` (`temperature`, `top_p`, …) is between you and the configured provider.
- **`ToolCallDelta` isn't a top-level export** — import it from `arvel_ai.contracts`.
- **Embeddings are per-driver** — a provider without an embeddings endpoint raises
  `AiCapabilityError`.

## See also

- [MCP Server](mcp.md) — expose your app's functions to AI agents, with auth.
- [Workflows](workflows.md) — durable, multi-step, signal-driven AI work.
- [Configuration](configuration.md) — every `config("ai")` key and default.
