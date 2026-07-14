"""AiServiceProvider — auto-registered via the ``arvel.providers`` entry point."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from arvel.kernel import ServiceProvider

from .commands import cli
from .config import DEFAULTS
from .manager import AiManager


class AiServiceProvider(ServiceProvider):
    def register(self) -> None:
        # Package config defaults — the host app's own values win on conflict.
        self.merge_config_from(DEFAULTS, "ai")
        self.app.singleton("ai", lambda c: AiManager(self.app))

    def boot(self) -> None:
        self.commands(cli)

        # MCP server: opt-in via config. Importing the tool modules runs their
        # @mcp_tool registrations; the route file serves /mcp + the RFC 9728
        # metadata document.
        config = self.app.make("config")
        if config.get("ai.mcp.enabled", False):
            for module in config.get("ai.mcp.tools", []) or []:
                import_module(str(module))
            self.load_routes_from(str(Path(__file__).parent / "routes.py"))
