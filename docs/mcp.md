# MCP Server

MCP (Model Context Protocol) is how AI agents — Claude, IDE assistants, autonomous tools — call into
applications. `arvel-ai` turns functions you mark with `@mcp_tool` into a spec-compliant MCP server
(protocol `2025-06-18`), behind real authentication. It is **off by default**; exposing your app to
agents is a deliberate act.

## Expose a tool

Decorate a function — sync or async — and drop it in `app/mcp_tools/`. Its signature becomes the
tool's input schema; every module in that folder is imported at boot, so the tool is live the
moment the file exists:

```python
# app/mcp_tools/order_status.py
from arvel_ai.mcp import mcp_tool

@mcp_tool(description="Look up an order's status by id")
async def order_status(order_id: int) -> str:
    order = await Order.find(order_id)
    return order.status.value
```

Turn the server on — no tool wiring needed:

```python
# config/ai.py
ai = {
    "mcp": {
        "enabled": True,
        "public_url": "https://shop.example.com",   # canonical https URL (required when enabled)
        "path": "/mcp",                              # where the endpoint mounts (default /mcp)
        # tools_dir: "app/mcp_tools" is the default — every *.py in it is autoloaded
        "auth": {"mode": "token", "token_env": "MCP_TOKEN"},
    },
}
```

The package mounts two routes for you — you don't register them in your app's `with_routing`
groups:

- `POST /mcp` — the JSON-RPC endpoint (`initialize`, `tools/list`, `tools/call`)
- `GET /.well-known/oauth-protected-resource` — the RFC 9728 metadata document clients use to find
  where to authenticate

Your tools aren't routes, so they don't live in `routes/`. They're plain functions registered by
the `@mcp_tool` decorator, and the server discovers them by **importing every `*.py` under
`app/mcp_tools/`** at boot — the same zero-wiring convention `config/` uses. One tool per file, or
group related ones; a leading-underscore file (`_helpers.py`) is skipped. Need a tool to live
elsewhere (a package, a shared module)? List it explicitly in `mcp.tools` and it's imported
alongside the folder.

Tool input schemas derive from your function signatures — scalars (`str`/`int`/`float`/`bool`),
`list[T]` (with item type), `dict`, and `Optional[T]`; a parameter with no default is `required`.
**Arguments are validated at the boundary** (required keys, types, no extras) before your function
runs.

## Authentication

Every `tools/list` and `tools/call` needs a bearer token. A missing/invalid one gets `401` with a
`WWW-Authenticate` challenge pointing at the metadata document.

### Token mode (dev/internal)

Clients send `Authorization: Bearer <token>`; the server compares it **constant-time** against the
env var named in `token_env`:

```python
"auth": {"mode": "token", "token_env": "MCP_TOKEN"}
```

```bash
# .env
MCP_TOKEN=a-long-random-secret
```

### OIDC mode (production)

The server becomes an OAuth 2.1 *resource server*; any OIDC provider (Keycloak, Auth0, …) is the
authorization server. Needs `uv add 'arvel[jwt]'`.

```python
"auth": {
    "mode": "oidc",
    "issuer": "https://idp.example.com/realms/shop",
    # jwks_uri: defaults to <issuer>/protocol/openid-connect/certs
    # audience: defaults to public_url + path
}
```

An unauthenticated request returns `401` with
`WWW-Authenticate: Bearer resource_metadata="…/.well-known/oauth-protected-resource"` — that header
is what makes MCP clients (Claude, IDEs) show their **login button** and run the OAuth flow
themselves (discovery, dynamic client registration, PKCE all happen between the client and your
IdP). Tokens are verified for signature (RS256/ES256 only — `none`/HS confusion is rejected),
issuer, expiry, **and audience**: a token minted for any other service is refused (RFC 8707 audience
binding). JWKS keys are fetched off the event loop and cached across requests.

## Security model — read before enabling

- **Tool arguments are untrusted input.** The schema gate checks shape, not meaning — an agent can
  pass any order id it likes. Authorize inside the tool the way you would in a route handler.
- **Design tools least-privilege.** Expose `order_status(order_id)`, not `run_sql(query)`. Small,
  specific tools are auditable; a generic escape hatch hands the agent your database. Don't leak
  existence either — an undistinguished `"not found"` beats confirming which ids exist.
- **Tool output is prompt input.** Whatever your tool returns is read by an LLM on the client side —
  a tool that echoes user-generated content can carry a prompt injection to the calling agent.
  Return data, not instructions, and treat stored user content in outputs like rendered HTML.
- **No token passthrough.** Tools never see the caller's bearer token, and the gateway's upstream AI
  calls always use your app's own provider keys.
- **Rate-limit the route.** Put arvel's throttle middleware on `/mcp`; the package deliberately
  doesn't guess your limits.
- **Errors return the message only** — never a traceback.

## Testing

The server is transport-agnostic, so you drive its JSON-RPC directly with no HTTP stack:

```python
from arvel_ai.mcp import McpServer, ToolRegistry
from arvel_ai.settings import McpSettings

server = McpServer(registry=my_registry, settings=McpSettings(...))
await server.authenticate({"authorization": "Bearer <token>"})   # raises McpAuthError on bad auth
result = await server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                              "params": {"name": "order_status", "arguments": {"order_id": 42}}})
```

To assert a single tool in isolation, build a `ToolRegistry`, register the function, and call
`await registry.call(name, arguments)`.

## Common mistakes & gotchas

- **`public_url` is required when enabled** — the metadata document and the 401 challenge are built
  from it; a wrong value breaks client login silently.
- **OIDC needs `arvel[jwt]`** (pyjwt) for JWKS validation.
- **Tool file outside `app/mcp_tools/`** — it isn't autoloaded, so the decorator never runs and the
  tool never appears in `tools/list`. Move it into the folder, or list it in `mcp.tools`.
- **Testing against a real client:** `npx @modelcontextprotocol/inspector` speaks the same protocol
  — point it at `http://localhost:8000/mcp` with your dev token.

## See also

- [The Gateway](gateway.md) — the AI gateway this package is built on.
- [Configuration](configuration.md) — every `config("ai.mcp")` key and default.
