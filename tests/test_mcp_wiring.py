"""Provider wiring for the MCP server: disabled by default; enabling imports
tool modules and contributes the route file."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from arvel.kernel import Application

from arvel_ai.mcp import registry
from arvel_ai.provider import AiServiceProvider, _import_tools


def test_mcp_disabled_by_default(app: Application) -> None:
    from arvel_ai.settings import AiSettings

    assert AiSettings().mcp.enabled is False
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


def test_import_tools_autoloads_the_folder(tmp_path: Path) -> None:
    # a temp app/mcp_tools/ package whose module registers a tool on import
    (tmp_path / "app" / "mcp_tools").mkdir(parents=True)
    (tmp_path / "app" / "__init__.py").write_text("")
    (tmp_path / "app" / "mcp_tools" / "__init__.py").write_text("")
    (tmp_path / "app" / "mcp_tools" / "products.py").write_text(
        "from arvel_ai.mcp import mcp_tool\n"
        "@mcp_tool()\n"
        "async def probe_product(id: int) -> str:\n"
        "    return 'ok'\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        _import_tools(str(tmp_path), "app/mcp_tools", [])
        assert any(t["name"] == "probe_product" for t in registry.descriptors())
    finally:
        registry.remove("probe_product")
        sys.path.remove(str(tmp_path))
        for mod in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
            del sys.modules[mod]


def test_import_tools_skips_a_missing_folder() -> None:
    _import_tools("/nonexistent", "app/mcp_tools", [])  # must not raise
