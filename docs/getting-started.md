# Getting Started

This walks you from `uv add` to a working AI call and a passing test — end to end, no gaps.

## 1. Install

```bash
uv add arvel-ai
```

That's the whole installation. `arvel-ai` ships an `arvel.providers` entry point, so its
`AiServiceProvider` auto-registers with the host app — you don't touch `bootstrap/providers.py`. On
boot it binds the `ai` service, registers the AI gateway as a health-checked
resource, and (when enabled) mounts the MCP routes.

The default `openai_compatible` driver runs on arvel's own HTTP client, whose `httpx` engine is part
of arvel core — nothing extra to install. The any-llm driver is an optional extra — **one
extra named after your provider** installs the driver *and* that provider's SDK:

```bash
uv add 'arvel-ai[anthropic]'      # any-llm driver + the Anthropic SDK, in one extra
```

Supported provider extras (mirroring [any-llm](https://docs.mozilla.ai/any-llm/)'s own):

> `anthropic` · `atlascloud` · `azure` · `azureanthropic` · `azureopenai` · `bedrock` ·
> `cascadia` · `cerebras` · `cohere` · `dashscope` · `databricks` · `deepinfra` · `deepseek` ·
> `fireworks` · `gemini` · `github` · `gmi` · `groq` · `huggingface` · `inception` · `llama` ·
> `llamacpp` · `llamafile` · `lmstudio` · `minimax` · `mistral` · `moonshot` · `mzai` ·
> `nebius` · `neosantara` · `ollama` · `openai` · `openrouter` · `otari` · `perplexity` ·
> `portkey` · `qiniu` · `requesty` · `sagemaker` · `sambanova` · `telnyx` · `together` ·
> `vertexai` · `vertexaianthropic` · `vllm` · `voyage` · `watsonx` · `xai` · `zai`

Two escape hatches: `arvel-ai[any-llm]` installs the bare any-llm SDK with no provider SDK,
and `arvel-ai[all]` installs every provider.

## 2. Configure

Create (or edit) `config/ai.py`. The gateway speaks the OpenAI HTTP format, and both Anthropic and
OpenAI expose OpenAI-compatible endpoints — so point it straight at a provider, no proxy required:

```python
# config/ai.py
from arvel import env

ai = {
    "default": "openai_compatible",          # which driver AI.chat() dispatches to
    "models": {                              # aliases: your code says "fast"/"smart"
        "fast": "claude-haiku-4-5",
        "smart": "claude-opus-4-8",
    },
    "drivers": {
        "openai_compatible": {
            "base_url": env("AI_GATEWAY_URL", "https://api.anthropic.com/v1"),
            "api_key_env": "AI_API_KEY",     # the NAME of the env var holding the key
            "model": env("AI_MODEL_FAST", "claude-haiku-4-5"),
        },
    },
}
```

> **Set `default` explicitly, as above.** The package's *built-in* default driver is `any_llm`,
> which needs a provider extra (`uv add 'arvel-ai[anthropic]'`). A base install with no config
> would hit a missing-extra error on the first call; `"default": "openai_compatible"` uses the
> driver that ships with the base install.

Put the key in your environment — never in config:

```bash
# .env
AI_API_KEY=sk-...
```

> **Any OpenAI-compatible endpoint works:** a provider directly
> (`https://api.anthropic.com/v1`, `https://api.openai.com/v1`), a LiteLLM proxy, or a local
> vLLM/Ollama server. For the full any-llm provider matrix, install your provider's extra
> (`uv add 'arvel-ai[anthropic]'`, see the list above) and set `"default": "any_llm"` (model ids
> become `provider:model`, e.g. `anthropic:claude-haiku-4-5`) —
> see [Drivers](gateway.md#drivers--model-aliases).

## 3. Make a call

```python
from arvel_ai import AI

reply = await AI.chat("Write a one-line tagline for a standing desk", model="fast")
print(reply.text)                 # "Stand up for better work."
print(reply.usage.output_tokens)
```

`model="fast"` resolves through your `models` aliases to `claude-haiku-4-5`. From here, the
[gateway guide](gateway.md) covers streaming, structured output, tools, and the error taxonomy.

## 4. Confirm it's wired up

Boot the app and check the resource-startup log (or `GET /health`). The AI resource runs a **real**
probe:

- `ok` — the gateway is reachable and your key works.
- `failed` — a wrong/missing key or an unreachable gateway (it won't report a false `ok`).
- `degraded` — not configured yet (no `base_url`).

## 5. Test without the network

Never call a real provider from your test suite. The fake is a first-class driver — swap it in the
same way production swaps providers:

```python
from arvel_ai import AI

async def test_generates_a_tagline():
    fake = AI.fake()                                 # swap in the fake driver
    fake.replies = ["Stand up for better work."]     # script the reply(ies)
    reply = await AI.chat("tagline please")
    assert reply.text == "Stand up for better work."
    fake.assert_chatted("tagline")                   # some request mentioned it
    AI.clear_swapped()                               # restore (do this in an autouse fixture)
```

```python
# conftest.py
import pytest
from arvel_ai import AI

@pytest.fixture(autouse=True)
def _reset_ai():
    yield
    AI.clear_swapped()
```

## Next steps

- [The Gateway](gateway.md) — the full `chat` / `stream` / `structured` / `embed` surface, tools,
  drivers, errors, observability, and testing.
- [MCP Server](mcp.md) — expose your app's functions to AI agents.
- [Configuration](configuration.md) — every `config("ai")` key and its default.
