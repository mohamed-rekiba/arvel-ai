"""The MCP server: expose app functions to MCP clients, behind real auth.

    from arvel_ai.mcp import mcp_tool

    @mcp_tool(description="Look up an order")
    async def get_order(order_id: int) -> str: ...

Wire-up: set config `ai.mcp.enabled = True`, list the modules holding your
tools in `ai.mcp.tools`, pick an auth mode, and the provider serves
`POST <path>` (JSON-RPC, spec 2025-06-18) + `GET /.well-known/oauth-protected-resource`.

Auth (S2/threat-model requirements):
- `token`  — static bearer compared constant-time against the env var named in
  `auth.token_env`. Dev/internal.
- `oidc`   — JWT validated against the issuer's JWKS (signature, `iss`, `exp`,
  and **`aud` = this server's canonical resource URI**, RFC 8707). The 401
  challenge carries `resource_metadata` so clients render their login flow.

Tool arguments are validated at this trust boundary (required keys, types,
no extras) BEFORE the tool runs. Tools never see the caller's token.
"""

from __future__ import annotations

import hmac
import inspect
import os
import types
from collections.abc import Callable, Mapping
from typing import Any, Union, cast, get_args, get_origin

import anyio
import anyio.to_thread

from .settings import McpAuthSettings, McpSettings

PROTOCOL_VERSION = "2025-06-18"

_SCALAR_JSON: dict[Any, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}
_PY_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _schema_for(annotation: Any) -> dict[str, Any]:
    """Turn a Python parameter annotation into a JSON-Schema node. Handles scalars, ``list[T]``
    (with its ``items``), ``dict`` (an object), and ``Optional[T]`` / ``T | None`` (the non-None
    side); anything else, including an un-annotated parameter, falls back to ``"string"``."""
    origin = get_origin(annotation)
    if origin is types.UnionType or origin is Union:  # Optional[T] / T | None
        arms = [a for a in get_args(annotation) if a is not type(None)]
        if len(arms) == 1:
            return _schema_for(arms[0])
        # a real multi-type union (int | str): leave it unconstrained rather than pick one arm
        # and reject the others
        return {}
    if annotation in _SCALAR_JSON:
        return {"type": _SCALAR_JSON[annotation]}
    if annotation is list or origin is list:
        args = get_args(annotation)
        return {"type": "array", "items": _schema_for(args[0]) if args else {"type": "string"}}
    if annotation is dict or origin is dict:
        return {"type": "object"}
    return {"type": "string"}


def _check_type(key: str, value: Any, schema: dict[str, Any]) -> None:
    """Validate ``value`` against a ``_schema_for`` node (top-level type + array items)."""
    expected = schema.get("type")
    if expected is None:  # an unconstrained schema (e.g. a multi-type union) — accept anything
        return
    py_type = _PY_TYPES[expected]
    # bool is an int subclass — never let True/False satisfy integer/number
    if isinstance(value, bool) and expected != "boolean":
        raise ValueError(f"argument {key!r}: expected {expected}")
    if not isinstance(value, py_type):
        raise ValueError(f"argument {key!r}: expected {expected}")
    if expected == "array" and "items" in schema and isinstance(value, list):
        # isinstance narrows value to list[Unknown]; cast to list[object] (not list[Any], which
        # mypy would flag as a redundant cast here) since the element type is only known once
        # _check_type validates each item below
        for item in cast(list[object], value):
            _check_type(key, item, schema["items"])


class McpAuthError(Exception):
    def __init__(self, status: int, message: str, www_authenticate: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.www_authenticate = www_authenticate


class ToolRegistry:
    """Named tools + JSON-Schema descriptors derived from function signatures."""

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    def tool(
        self, name: str | None = None, description: str | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def register(fn: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or fn.__name__
            properties: dict[str, Any] = {}
            required: list[str] = []
            for param in inspect.signature(fn, eval_str=True).parameters.values():
                properties[param.name] = _schema_for(param.annotation)
                if param.default is inspect.Parameter.empty:
                    required.append(param.name)
            self._tools[tool_name] = {
                "fn": fn,
                "descriptor": {
                    "name": tool_name,
                    "description": description or (fn.__doc__ or "").strip(),
                    "inputSchema": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
            return fn

        return register

    def remove(self, name: str) -> None:
        self._tools.pop(name, None)

    def descriptors(self) -> list[dict[str, Any]]:
        return [entry["descriptor"] for entry in self._tools.values()]

    async def call(self, name: str, arguments: Mapping[str, Any]) -> Any:
        entry = self._tools.get(name)
        if entry is None:
            raise ValueError(f"unknown tool {name!r}")
        schema = entry["descriptor"]["inputSchema"]
        properties: dict[str, Any] = schema["properties"]
        for key in schema["required"]:
            if key not in arguments:
                raise ValueError(f"missing required argument {key!r}")
        for key, value in arguments.items():
            if key not in properties:
                raise ValueError(f"unexpected argument {key!r}")
            _check_type(key, value, properties[key])
        result = entry["fn"](**dict(arguments))
        if inspect.isawaitable(result):
            result = await result
        return result


#: The default registry `@mcp_tool` writes into; apps import this decorator.
registry = ToolRegistry()


def mcp_tool(
    name: str | None = None, description: str | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return registry.tool(name=name, description=description)


class McpServer:
    """Transport-agnostic core: authenticate headers, handle JSON-RPC payloads.

    The HTTP glue (routes.py) stays a thin adapter so this logic is fully
    testable without an HTTP stack.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        settings: McpSettings | None = None,
        server_name: str = "arvel-ai",
    ) -> None:
        self.registry = registry
        self.settings = settings if settings is not None else McpSettings()
        self.server_name = server_name
        # one PyJWKClient per jwks_uri, so its signing-key cache survives across requests
        self._jwk_clients: dict[str, Any] = {}

    # -- auth -------------------------------------------------------------------

    @property
    def _auth(self) -> McpAuthSettings:
        return self.settings.auth

    @property
    def resource_uri(self) -> str:
        public = (self.settings.public_url or "").rstrip("/")
        return f"{public}{self.settings.path}"

    def _challenge(self, message: str, status: int = 401) -> McpAuthError:
        public = (self.settings.public_url or "").rstrip("/")
        header = f'Bearer resource_metadata="{public}/.well-known/oauth-protected-resource"'
        return McpAuthError(status, message, www_authenticate=header)

    async def authenticate(self, headers: Mapping[str, str]) -> None:
        """Raise McpAuthError unless the request carries a valid bearer token."""
        authorization = headers.get("authorization") or headers.get("Authorization") or ""
        if not authorization.startswith("Bearer "):
            raise self._challenge("missing bearer token")
        token = authorization[len("Bearer ") :]
        mode = self._auth.mode
        if mode == "token":
            expected = os.environ.get(self._auth.token_env, "")
            if not expected or not hmac.compare_digest(token, expected):
                raise self._challenge("invalid token")
            return
        if mode == "oidc":
            await self._authenticate_oidc(token)
            return
        raise self._challenge(f"unknown auth mode {mode!r}")

    def _jwk_client(self, jwks_uri: str) -> Any:
        """The cached PyJWKClient for this jwks_uri (built lazily), so we don't refetch the key
        set on every request."""
        client = self._jwk_clients.get(jwks_uri)
        if client is None:
            import jwt as pyjwt  # arvel's jwt extra

            client = pyjwt.PyJWKClient(jwks_uri)
            self._jwk_clients[jwks_uri] = client
        return client

    async def _authenticate_oidc(self, token: str) -> None:
        issuer = self._auth.issuer
        if not issuer:
            raise self._challenge("oidc auth is not configured (set ai.mcp.auth.issuer)")
        if not self._auth.audience and not self.settings.public_url:
            # audience would degrade to a bare path — refuse rather than validate against a
            # weak/foot-gun aud
            raise self._challenge("oidc auth needs public_url (or an explicit auth.audience)")
        jwks_uri = self._auth.jwks_uri or f"{issuer.rstrip('/')}/protocol/openid-connect/certs"
        audience = self._auth.audience or self.resource_uri
        try:
            client = self._jwk_client(jwks_uri)
        except ImportError as exc:  # pragma: no cover
            raise self._challenge("oidc auth needs pyjwt: uv add 'arvel[jwt]'") from exc
        # PyJWKClient.get_signing_key_from_jwt + decode are blocking urllib I/O — run them off
        # the event loop so an OIDC-protected server doesn't head-of-line-block on the JWKS fetch.
        try:
            await anyio.to_thread.run_sync(self._verify_token, client, token, audience, issuer)
        except Exception as exc:
            raise self._challenge(f"token rejected: {type(exc).__name__}") from exc

    @staticmethod
    def _verify_token(client: Any, token: str, audience: str, issuer: str) -> None:
        import jwt as pyjwt

        key = client.get_signing_key_from_jwt(token).key
        pyjwt.decode(
            token,
            key=key,
            algorithms=["RS256", "ES256"],
            issuer=issuer,
            audience=audience,  # RFC 8707 audience binding — non-negotiable
            # require the claims we validate — a token minted without exp would never expire
            options={"require": ["exp", "iss", "aud"]},
        )

    def protected_resource_metadata(self) -> dict[str, Any]:
        return {
            "resource": self.resource_uri,
            "authorization_servers": [self._auth.issuer] if self._auth.issuer else [],
            "bearer_methods_supported": ["header"],
        }

    # -- JSON-RPC ------------------------------------------------------------------

    async def handle(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        method = payload.get("method", "")
        request_id = payload.get("id")
        if method.startswith("notifications/"):
            return None
        if method == "initialize":
            return self._result(
                request_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": self.server_name, "version": "1"},
                },
            )
        if method == "ping":
            return self._result(request_id, {})
        if method == "tools/list":
            return self._result(request_id, {"tools": self.registry.descriptors()})
        if method == "tools/call":
            params: dict[str, Any] = payload.get("params") or {}
            try:
                value = await self.registry.call(
                    params.get("name", ""), params.get("arguments") or {}
                )
                content = [{"type": "text", "text": str(value)}]
                return self._result(request_id, {"content": content, "isError": False})
            except Exception as exc:  # message only — never a traceback (threat model)
                return self._result(
                    request_id,
                    {"content": [{"type": "text", "text": str(exc)}], "isError": True},
                )
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"method not found: {method}"},
        }

    @staticmethod
    def _result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
