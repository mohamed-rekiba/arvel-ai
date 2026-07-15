"""MCP server: JSON-RPC protocol, tool registry + boundary validation, and the
two auth modes (token and oidc, with their metadata + 401 challenge shapes)."""

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


def test_registry_maps_non_scalar_annotations() -> None:
    reg = ToolRegistry()

    @reg.tool()
    async def search(tags: list[str], filters: dict[str, int], limit: int | None = None) -> str:
        return "ok"

    props = {t["name"]: t for t in reg.descriptors()}["search"]["inputSchema"]["properties"]
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert props["filters"] == {"type": "object"}
    assert props["limit"] == {"type": "integer"}  # Optional[int] -> the non-None arm


async def test_union_typed_arg_is_unconstrained_and_accepts_either_type() -> None:
    reg = ToolRegistry()

    @reg.tool()
    async def fetch(ref: int | str) -> str:
        return str(ref)

    # a genuine multi-type union is left unconstrained rather than pinned to one arm
    props = {t["name"]: t for t in reg.descriptors()}["fetch"]["inputSchema"]["properties"]
    assert props["ref"] == {}
    assert await reg.call("fetch", {"ref": 5}) == "5"
    assert await reg.call("fetch", {"ref": "abc"}) == "abc"


async def test_call_validates_array_and_object_arguments() -> None:
    reg = ToolRegistry()

    @reg.tool()
    async def search(tags: list[str], filters: dict[str, int]) -> int:
        return len(tags) + len(filters)

    assert await reg.call("search", {"tags": ["a", "b"], "filters": {"x": 1}}) == 3
    with pytest.raises(ValueError, match="expected array"):
        await reg.call("search", {"tags": "not-a-list", "filters": {}})
    with pytest.raises(ValueError, match="expected string"):  # bad array item
        await reg.call("search", {"tags": [1, 2], "filters": {}})
    with pytest.raises(ValueError, match="expected object"):
        await reg.call("search", {"tags": [], "filters": []})


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


async def test_token_mode_accepts_matching_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_TOKEN", "s3cret")
    server = McpServer(registry=make_registry(), settings=auth_settings("token"))
    await server.authenticate({"authorization": "Bearer s3cret"})  # no raise


@pytest.mark.parametrize(
    "header", [{}, {"authorization": "Bearer wrong"}, {"authorization": "Basic x"}]
)
async def test_token_mode_rejects_with_challenge(
    monkeypatch: pytest.MonkeyPatch, header: dict[str, str]
) -> None:
    monkeypatch.setenv("MCP_TOKEN", "s3cret")
    server = McpServer(registry=make_registry(), settings=auth_settings("token"))
    with pytest.raises(McpAuthError) as exc:
        await server.authenticate(header)
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


async def test_oidc_mode_requires_issuer_config() -> None:
    server = McpServer(registry=make_registry(), settings=auth_settings("oidc"))
    with pytest.raises(McpAuthError) as exc:
        await server.authenticate({"authorization": "Bearer sometoken"})
    assert exc.value.status == 401


async def test_oidc_verifies_signature_audience_and_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mint real RS256 tokens against a stubbed JWKS and confirm the OIDC path accepts a valid
    one and rejects wrong-audience (RFC 8707), expired, and disallowed-algorithm tokens."""
    import time

    pyjwt = pytest.importorskip("jwt")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_key = private_key.public_key()

    issuer = "https://idp.example.com/realms/app"
    audience = "https://app.example.com/mcp"  # == resource_uri (public_url + path)
    server = McpServer(registry=make_registry(), settings=auth_settings("oidc", issuer=issuer))

    # stub the JWKS client so nothing hits the network — hand back our public key
    class _FakeKey:
        key = public_key

    class _FakeJwkClient:
        def get_signing_key_from_jwt(self, _token: str) -> Any:
            return _FakeKey()

    monkeypatch.setattr(server, "_jwk_client", lambda _uri: _FakeJwkClient())

    def mint(*, aud: str = audience, ttl: int = 3600) -> str:
        payload = {"iss": issuer, "aud": aud, "sub": "u1", "exp": int(time.time()) + ttl}
        return pyjwt.encode(payload, private_pem, algorithm="RS256")

    async def rejected(token: str) -> None:
        with pytest.raises(McpAuthError) as exc:
            await server.authenticate({"authorization": f"Bearer {token}"})
        assert exc.value.status == 401

    # a correctly-signed, in-audience, unexpired token passes
    await server.authenticate({"authorization": f"Bearer {mint()}"})
    # wrong audience, expired, and a disallowed algorithm (HS256) each get a 401
    await rejected(mint(aud="https://evil.example.com/mcp"))
    await rejected(mint(ttl=-10))
    await rejected(
        pyjwt.encode(
            {"iss": issuer, "aud": audience, "exp": int(time.time()) + 3600},
            "a" * 32,  # length is irrelevant; the token is rejected on algorithm, not key
            algorithm="HS256",
        )
    )
