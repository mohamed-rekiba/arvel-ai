"""Provider wiring for the MCP server: disabled by default; enabling imports
tool modules and contributes the route file."""

from __future__ import annotations

import pytest

from arvel.kernel import Application

from arvel_ai.mcp import registry
from arvel_ai.provider import AiServiceProvider


def test_mcp_disabled_by_default(app: Application) -> None:
    assert app.make("config").get("ai.mcp.enabled") is False
    assert not [p for p in app.route_files if "arvel_ai" in str(p)]


def test_enabling_mcp_wires_routes_and_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    application = Application()
    # host-app config wins over package defaults, and MUST exist before
    # register(): route wiring happens there (boot runs too late for routes)
    application.make("config").set(
        "ai",
        {"mcp": {"enabled": True, "tools": ["sample_mcp_tools"], "path": "/mcp"}},
    )
    provider = AiServiceProvider(application)
    provider.register()
    provider.boot()
    try:
        assert any(str(p).endswith("arvel_ai/routes.py") for p in application.route_files)
        assert any(t["name"] == "sample_lookup" for t in registry.descriptors())
    finally:
        registry.remove("sample_lookup")
