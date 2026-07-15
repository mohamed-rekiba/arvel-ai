# Configuration

Everything lives under `config("ai")` — a typed, validated view (`AiSettings`, built on msgspec).
The defaults below **are** the package defaults; you set only what you want to change, and host-app
values always win. Keys and tokens are never stored here: config holds the *name* of the env var
that holds the secret (`api_key_env`, `token_env`), and the value is read from the environment at
call time.

```python
# config/ai.py
ai = {
    "default": "litellm",
    "models": {},
    "include_raw": False,
    "critical": False,
    "drivers": {...},
    "mcp": {...},
    "workflows": {...},
}
```

## Top level

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `default` | str | `"litellm"` | which driver `AI.chat()` dispatches to (`litellm` / `openai_compatible` / a custom name) |
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

## `drivers.litellm`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `model` | str \| None | `None` | default model, LiteLLM form (e.g. `anthropic/claude-haiku-4-5`) |
| `timeout` | float | `60.0` | per-request timeout |
| `max_retries` | int | `2` | LiteLLM's own retry count |
| `include_raw` | bool | `False` | attach raw provider JSON |

Provider keys use each provider's standard env var (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …).

## `mcp`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `enabled` | bool | `False` | mount the MCP server (off by default — exposing your app is deliberate) |
| `path` | str | `"/mcp"` | where the JSON-RPC endpoint mounts |
| `public_url` | str \| None | `None` | canonical https URL; **required** when enabled |
| `tools` | list[str] | `[]` | modules to import at boot so their `@mcp_tool`s register |
| `auth` | table | — | see below |

### `mcp.auth`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `mode` | str | `"token"` | `"token"` (shared secret) or `"oidc"` (JWT) |
| `token_env` | str | `"MCP_TOKEN"` | **name** of the env var holding the bearer token (token mode) |
| `issuer` | str \| None | `None` | OIDC issuer URL (oidc mode) |
| `jwks_uri` | str \| None | `None` | JWKS URL; defaults to the issuer's standard certs URL |
| `audience` | str \| None | `None` | expected `aud`; defaults to `public_url + path` |

## `workflows`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `default` | str | `"queue"` | `"queue"` / `"temporal"` / `"fake"` |
| `drivers.temporal.target` | str | `"localhost:7233"` | Temporal frontend address |
| `drivers.temporal.namespace` | str | `"default"` | Temporal namespace |
| `drivers.temporal.task_queue` | str | `"arvel-ai"` | Temporal task queue |

## Reading config yourself

`AiSettings` auto-loads and validates on instantiation, so you rarely touch it directly — the
manager does. When you need a value:

```python
from arvel_ai.settings import AiSettings

settings = AiSettings()          # reads + validates config("ai")
settings.default                 # "litellm"
settings.mcp.enabled             # False
```

## Common mistakes & gotchas

- **The built-in `default` is `litellm`**, which needs `uv add 'arvel-ai[litellm]'`. Set
  `"default": "openai_compatible"` to use the base-install driver (see
  [Getting Started](getting-started.md)).
- **Secrets are env-var *names* here, not values** — `api_key_env`, `token_env`.
- **`mcp.public_url` is required when `mcp.enabled`** — the metadata document and 401 challenge are
  built from it.

## See also

- [The Gateway](gateway.md) · [MCP Server](mcp.md) · [Workflows](workflows.md).
