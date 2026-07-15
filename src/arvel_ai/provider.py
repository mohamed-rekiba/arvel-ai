"""AiServiceProvider — auto-registered via the ``arvel.providers`` entry point."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from arvel.kernel import ServiceProvider

from .commands import cli
from .manager import AiManager
from .settings import AiSettings
from .workflows.manager import WorkflowManager


class AiServiceProvider(ServiceProvider):
    def register(self) -> None:
        # Config defaults are the typed AiSettings field defaults (the framework
        # Settings pattern) — no merge_config_from, no DEFAULTS dict; the host
        # app's config("ai") section overrides them.
        self.app.singleton("ai", lambda c: AiManager(self.app))
        self.app.singleton("ai.workflows", lambda c: WorkflowManager(self.app))

        # MCP server: opt-in via config. This lives in register() DELIBERATELY:
        # route files load right after provider registration, while the async
        # provider boot loop runs later (ASGI lifespan) — a boot()-time
        # load_routes_from is too late for a package.
        mcp = self._settings().mcp
        if mcp.enabled:
            for module in mcp.tools:
                import_module(module)
            self.load_routes_from(str(Path(__file__).parent / "routes.py"))

    def _settings(self) -> AiSettings:
        # scope to THIS app's config section (like Manager(app) does), not global
        return AiSettings.from_source(self.app.config("ai"))

    def boot(self) -> None:
        self.commands(cli)

        # Register the AI gateway as a health-checkable, drained-at-shutdown
        # resource (DR-0039) — it appears in the resource-startup log and /health.
        from .resource import AiResource

        self.app.resources.register(
            AiResource(self.app.make("ai"), critical=self._settings().critical)
        )
