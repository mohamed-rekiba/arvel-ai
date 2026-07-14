"""AiServiceProvider — auto-registered via the ``arvel.providers`` entry point."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from arvel.kernel import ServiceProvider

from .commands import cli
from .config import DEFAULTS
from .manager import AiManager
from .workflows.manager import WorkflowManager


class AiServiceProvider(ServiceProvider):
    def register(self) -> None:
        # Package config defaults — the host app's own values win on conflict.
        self.merge_config_from(DEFAULTS, "ai")
        self.app.singleton("ai", lambda c: AiManager(self.app))
        self.app.singleton("ai.workflows", lambda c: WorkflowManager(self.app))

        # MCP server: opt-in via config. This lives in register() DELIBERATELY:
        # route files load right after provider registration, while the async
        # provider boot loop runs later (ASGI lifespan) — a boot()-time
        # load_routes_from is too late for a package.
        config = self.app.make("config")
        if config.get("ai.mcp.enabled", False):
            for module in config.get("ai.mcp.tools", []) or []:
                import_module(str(module))
            self.load_routes_from(str(Path(__file__).parent / "routes.py"))

    def boot(self) -> None:
        self.commands(cli)

        # Register the AI gateway as a health-checkable, drained-at-shutdown
        # resource (DR-0039) — it appears in the resource-startup log and /health.
        from .resource import AiResource

        critical = bool(self.app.make("config").get("ai.critical", False))
        self.app.resources.register(AiResource(self.app.make("ai"), critical=critical))
