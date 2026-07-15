"""MCP server: JSON-RPC protocol, tool registry + boundary validation, and the
two auth modes (token / oidc metadata + 401 challenge shapes per S2)."""

from __future__ import annotations

from typing import Any

import pytest

from arvel_ai.settings import McpAuthSettings, McpSettings
from arvel_ai.mcp import (
    McpAuthError,
    McpServer,
    ToolRegistry,
    mcp_tool,
    registry as global_registry,
)

# ---- tool registry ----------------------------------------------------------


def make_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(description="Add two numbers")
    def add(a: int, b: int) -> int:
        return a + b

    @reg.tool(name="greet", description="Greet someone")
    async def hello(name: str, excited: bool = False) -> str:
        return f"hi {name}{'!' if excited else ''}"

    return reg


def test_registry_derives_schemas_from_signatures() -> None:
    reg = make_registry()
    tools = {t["name"]: t for t in reg.descriptors()}
    assert tools["add"]["inputSchema"]["properties"]["a"] == {"type": "integer"}
    assert tools["add"]["inputSchema"]["required"] == ["a", "b"]
    assert tools["greet"]["inputSchema"]["required"] == ["name"]  # default -> optional


async def test_call_validates_arguments_at_the_boundary() -> None:
    reg = make_registry()
    assert await reg.call("add", {"a": 1, "b": 2}) == 3
    with pytest.raises(ValueError, match="missing required"):
        await reg.call("add", {"a": 1})
    with pytest.raises(ValueError, match="expected integer"):
        await reg.call("add", {"a": "one", "b": 2})
    with pytest.raises(ValueError, match="unknown tool"):
        await reg.call("nope", {})
    with pytest.raises(ValueError, match="unexpected argument"):
        await reg.call("add", {"a": 1, "b": 2, "c": 3})


def test_global_decorator_registers() -> None:
    @mcp_tool(description="probe")
    def probe() -> str:
        return "ok"

    try:
        assert any(t["name"] == "probe" for t in global_registry.descriptors())
    finally:
        global_registry.remove("probe")


# ---- JSON-RPC protocol -------------------------------------------------------


@pytest.fixture()
def server() -> McpServer:
    return McpServer(registry=make_registry(), server_name="test-app")


async def test_initialize_and_tools_list(server: McpServer) -> None:
    init = await server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init is not None
    assert init["result"]["serverInfo"]["name"] == "test-app"
    assert "tools" in init["result"]["capabilities"]

    assert await server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    listed = await server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert listed is not None
    assert {t["name"] for t in listed["result"]["tools"]} == {"add", "greet"}


async def test_tools_call_success_and_tool_error(server: McpServer) -> None:
    ok = await server.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "add", "arguments": {"a": 2, "b": 3}},
        }
    )
    assert ok is not None
    assert ok["result"]["content"] == [{"type": "text", "text": "5"}]
    assert ok["result"]["isError"] is False

    bad = await server.handle(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "add", "arguments": {"a": 1}},
        }
    )
    assert bad is not None
    assert bad["result"]["isError"] is True
    assert "missing required" in bad["result"]["content"][0]["text"]
    # no internals leak: just the message, no traceback
    assert "Traceback" not in bad["result"]["content"][0]["text"]


async def test_unknown_method_is_jsonrpc_error(server: McpServer) -> None:
    out = await server.handle({"jsonrpc": "2.0", "id": 9, "method": "bogus/method"})
    assert out is not None
    assert out["error"]["code"] == -32601


# ---- auth ---------------------------------------------------------------------


def auth_settings(mode: str, **extra: Any) -> McpSettings:
    return McpSettings(
        enabled=True,
        path="/mcp",
        public_url="https://app.example.com",
        auth=McpAuthSettings(mode=mode, token_env="MCP_TOKEN", **extra),
    )


def test_token_mode_accepts_matching_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_TOKEN", "s3cret")
    server = McpServer(registry=make_registry(), settings=auth_settings("token"))
    server.authenticate({"authorization": "Bearer s3cret"})  # no raise


@pytest.mark.parametrize(
    "header", [{}, {"authorization": "Bearer wrong"}, {"authorization": "Basic x"}]
)
def test_token_mode_rejects_with_challenge(
    monkeypatch: pytest.MonkeyPatch, header: dict[str, str]
) -> None:
    monkeypatch.setenv("MCP_TOKEN", "s3cret")
    server = McpServer(registry=make_registry(), settings=auth_settings("token"))
    with pytest.raises(McpAuthError) as exc:
        server.authenticate(header)
    assert exc.value.status == 401
    # the S2 challenge shape that makes clients show the login button
    assert (
        exc.value.www_authenticate
        == 'Bearer resource_metadata="https://app.example.com/.well-known/oauth-protected-resource"'
    )


def test_protected_resource_metadata_document() -> None:
    server = McpServer(
        registry=make_registry(),
        settings=auth_settings("oidc", issuer="https://idp.example.com/realms/app"),
    )
    doc = server.protected_resource_metadata()
    assert doc == {
        "resource": "https://app.example.com/mcp",
        "authorization_servers": ["https://idp.example.com/realms/app"],
        "bearer_methods_supported": ["header"],
    }


def test_oidc_mode_requires_issuer_config() -> None:
    server = McpServer(registry=make_registry(), settings=auth_settings("oidc"))
    with pytest.raises(McpAuthError) as exc:
        server.authenticate({"authorization": "Bearer sometoken"})
    assert exc.value.status == 401
