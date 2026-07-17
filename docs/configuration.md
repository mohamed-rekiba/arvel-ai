# Configuration

Everything lives under `config("ai")` — a typed, validated view (`AiSettings`, built on msgspec).
The defaults below **are** the package defaults; you set only what you want to change, and host-app
values always win. Keys and tokens are never stored here: config holds the *name* of the env var
that holds the secret (`api_key_env`, `token_env`), and the value is read from the environment at
call time.

```python
# config/ai.py
ai = {
    "default": "any_llm",
    "models": {},
    "include_raw": False,
    "critical": False,
    "drivers": {...},
    "mcp": {...},
}
```

## Top level

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `default` | str | `"any_llm"` | which driver `AI.chat()` dispatches to (`any_llm` / `openai_compatible` / a custom name) |
| `models` | dict[str, str] | `{}` | aliases → concrete model ids; `AI.chat(..., model="fast")` resolves here |
| `include_raw` | bool | `False` | attach the provider's raw JSON to responses (debugging) |
| `critical` | bool | `False` | does an AI outage abort app boot? `False` = degrade instead |

## `drivers.openai_compatible`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `base_url` | str \| None | `None` | the OpenAI-compatible endpoint; **required** for real calls |
| `api_key_env` | str | `"AI_API_KEY"` | **name** of the env var holding the bearer key |
| `model` | str \| None | `None` | default model when a call names none |
| `timeout` | float | `60.0` | per-request timeout (seconds) |
| `include_raw` | bool | `False` | attach raw provider JSON |

## `drivers.any_llm`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `model` | str \| None | `None` | default model, any-llm form — `provider:model`, colon-separated (e.g. `anthropic:claude-haiku-4-5`) |
| `timeout` | float | `60.0` | per-request timeout (driver-enforced; per-chunk while streaming) |
| `max_retries` | int | `2` | driver-level retries on retryable errors (rate limit / timeout / 5xx) |
| `include_raw` | bool | `False` | attach raw provider JSON |

Provider keys use each provider's standard env var (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …).

## `mcp`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `enabled` | bool | `False` | mount the MCP server (off by default — exposing your app is deliberate) |
| `path` | str | `"/mcp"` | where the JSON-RPC endpoint mounts |
| `public_url` | str \| None | `None` | canonical https URL; **required** when enabled |
| `tools_dir` | str | `"app/mcp_tools"` | folder autoloaded at boot — every `*.py` in it registers its `@mcp_tool`s |
| `tools` | list[str] | `[]` | extra modules to import (override) for tools living outside `tools_dir` |
| `auth` | table | — | see below |

### `mcp.auth`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `mode` | str | `"token"` | `"token"` (shared secret) or `"oidc"` (JWT) |
| `token_env` | str | `"MCP_TOKEN"` | **name** of the env var holding the bearer token (token mode) |
| `issuer` | str \| None | `None` | OIDC issuer URL (oidc mode) |
| `jwks_uri` | str \| None | `None` | JWKS URL; defaults to the issuer's standard certs URL |
| `audience` | str \| None | `None` | expected `aud`; defaults to `public_url + path` |

## Reading config yourself

`AiSettings` auto-loads and validates on instantiation, so you rarely touch it directly — the
manager does. When you need a value:

```python
from arvel_ai.settings import AiSettings

settings = AiSettings()          # reads + validates config("ai")
settings.default                 # "any_llm"
settings.mcp.enabled             # False
```

## Common mistakes & gotchas

- **The built-in `default` is `any_llm`**, which needs a provider extra — one extra installs the
  driver and that provider's SDK, e.g. `uv add 'arvel-ai[anthropic]'` (full list in
  [Getting Started](getting-started.md)). Set `"default": "openai_compatible"` to use the
  base-install driver instead.
- **any-llm model ids are `provider:model`** (colon, not slash) — `anthropic:claude-haiku-4-5` —
  and the provider prefix must match the extra you installed.
- **Secrets are env-var *names* here, not values** — `api_key_env`, `token_env`.
- **`mcp.public_url` is required when `mcp.enabled`** — the metadata document and 401 challenge are
  built from it.

## See also

- [The Gateway](gateway.md) · [MCP Server](mcp.md).
