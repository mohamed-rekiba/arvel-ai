"""MCP HTTP routes — loaded by the provider only when `ai.mcp.enabled` is true.

Thin adapter over arvel_ai.mcp.McpServer: auth challenge -> 401 + WWW-Authenticate
(the header that makes MCP clients render their login flow), JSON-RPC pass-through,
and the RFC 9728 protected-resource metadata document.
"""

from __future__ import annotations

from typing import Any

from arvel import Route
from arvel.http.response import Response
from arvel.kernel.globals import app

from .mcp import McpAuthError, McpServer, registry


def _server() -> McpServer:
    config = app("config")
    return McpServer(
        registry=registry,
        config=config.get("ai.mcp", {}) or {},
        server_name=str(config.get("app.name", "arvel-app")),
    )


async def mcp_endpoint(request: Any) -> Response:
    server = _server()
    try:
        server.authenticate(request.headers)
    except McpAuthError as exc:
        headers = {"WWW-Authenticate": exc.www_authenticate} if exc.www_authenticate else {}
        return Response(
            content={"error": str(exc)}, status=exc.status, headers=headers
        )
    result = await server.handle(await request.json() or {})
    if result is None:  # notification — acknowledged, no body
        return Response(content=None, status=202)
    return Response(content=result, status=200)


async def resource_metadata(request: Any) -> dict[str, Any]:
    return _server().protected_resource_metadata()


Route.post(str(app("config").get("ai.mcp.path", "/mcp")), mcp_endpoint, name="ai.mcp")
Route.get(
    "/.well-known/oauth-protected-resource", resource_metadata, name="ai.mcp.metadata"
)
