# arvel-ai

**One stable API over many AI providers — plus a secured MCP server and durable workflows — for
the [arvel](https://pypi.org/project/arvel/) framework.**

Your app talks to one contract: `AI.chat`, `AI.stream`, `AI.structured`, `AI.embed`. Which provider
actually serves the request — Anthropic, OpenAI, a LiteLLM proxy, a local vLLM or Ollama — is a
config choice, not a code change. No provider SDK type (`httpx`, `litellm`, `temporalio`) ever
crosses into your code; the boundary is enforced by the import-linter, not just intended.

```bash
uv add arvel-ai                 # gateway (openai_compatible + fake) + MCP server
uv add 'arvel-ai[litellm]'      # + the LiteLLM driver: 100+ providers
uv add 'arvel-ai[temporal]'     # + the Temporal workflow driver
```

Installing registers the provider automatically — `app.make("ai")` and the `AI` facade work with
zero wiring.

```python
from arvel_ai import AI

reply = await AI.chat("Summarize this review in one line", model="fast")
print(reply.text)

async for delta in AI.stream("Write a product description"):
    ...                                        # TextDelta / ToolCallDelta / StreamEnd

copy = await AI.structured(ProductCopy, "Write copy for red wool socks")
#      a typed, validated ProductCopy — not a dict
```

## What's in the box

- **The gateway** — `chat` / `stream` (SSE) / `structured` (typed output) / `embed`, over a stable
  msgspec contract and one `AiError` taxonomy. Swap providers in config; your code never changes.
- **Model aliases** — code says `model="fast"`/`"smart"`; ops maps those to real ids, so a retired
  model is a one-line config edit, not a code hunt.
- **A secured MCP server** — expose your app's functions to AI agents over the Model Context
  Protocol, with token or OIDC auth (RFC 8707 audience binding). Off by default.
- **Durable workflows** — `Workflow.start`/`.signal`/`.status`, on a queue by default or **Temporal**
  for retry-across-hours, human-in-the-loop flows.
- **First-class fakes** — `AI.fake()` / `Workflow.fake()` test AI code with no network, the same way
  you test mail.
- **A health-checked resource** — the gateway reports on `/health` and the startup log via a real
  probe: a wrong or missing key shows as `failed`, not a false `ok`.

## Point it at a provider

The gateway speaks the OpenAI HTTP format, and both Anthropic and OpenAI expose OpenAI-compatible
endpoints — so you need no proxy in front:

```python
# config/ai.py
from arvel import env

ai = {
    "default": "openai_compatible",
    "models": {"fast": "claude-haiku-4-5", "smart": "claude-opus-4-8"},
    "drivers": {
        "openai_compatible": {
            "base_url": env("AI_GATEWAY_URL", "https://api.anthropic.com/v1"),
            "api_key_env": "AI_API_KEY",       # the NAME of the env var, never the key itself
            "model": env("AI_MODEL_FAST", "claude-haiku-4-5"),
        },
    },
}
```

Keys live in environment variables only — config holds the env var *name*, never the secret.

## Documentation

| Guide | What's in it |
|-------|--------------|
| [Getting Started](docs/getting-started.md) | install → configure → first call → first test, end to end |
| [The Gateway](docs/gateway.md) | `chat`/`stream`/`structured`/`embed`, tools, drivers, aliases, errors, observability, testing |
| [MCP Server](docs/mcp.md) | expose tools to agents, token & OIDC auth, the security model |
| [Workflows](docs/workflows.md) | `@workflow`, `start`/`signal`/`status`, the queue vs Temporal engines |
| [Configuration](docs/configuration.md) | the complete `config("ai")` reference — every key and default |

## Requirements

- Python **3.14+**
- **arvel** (installed with the package)
- Optional engines: `arvel-ai[litellm]`, `arvel-ai[temporal]`; OIDC MCP auth needs `arvel[jwt]`

## License

MIT — see [LICENSE](LICENSE).
