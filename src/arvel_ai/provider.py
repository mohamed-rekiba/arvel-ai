"""AiServiceProvider — auto-registered via the ``arvel.providers`` entry point."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

from arvel.kernel import ServiceProvider

from .commands import cli
from .manager import AiManager
from .settings import AiSettings

if TYPE_CHECKING:
    from arvel.contracts import Container


def _import_tools(base_path: str, tools_dir: str, modules: list[str]) -> None:
    """Load the app's MCP tools so their ``@mcp_tool`` decorators run. Autoloads every ``*.py``
    under ``tools_dir`` (default ``app/mcp_tools/``, a convention folder like ``app/listeners``
    — DR-0045/0046), then imports any explicit ``modules`` from config as an override/addition.
    The folder is loaded **by path** (like arvel's ``config/*.py`` and listener discovery), so it
    doesn't depend on ``base_path`` being on ``sys.path``. Tools self-register when their module
    executes — no reflection needed."""
    directory = Path(base_path) / tools_dir
    if directory.is_dir():
        for file in sorted(directory.glob("*.py")):
            if file.stem.startswith("_"):
                continue
            spec = importlib.util.spec_from_file_location(f"_arvel_mcp_tool_{file.stem}", file)
            if spec is None or spec.loader is None:  # pragma: no cover - defensive
                continue
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
    for module in modules:
        importlib.import_module(module)


class AiServiceProvider(ServiceProvider):
    def register(self) -> None:
        # Config defaults are the typed AiSettings field defaults (the framework
        # Settings pattern) — no merge_config_from, no DEFAULTS dict; the host
        # app's config("ai") section overrides them.
        def make_ai(c: Container) -> AiManager:
            return AiManager(self.app)

        self.app.singleton("ai", make_ai)

        # MCP server: opt-in via config. This lives in register() DELIBERATELY:
        # route files load right after provider registration, while the async
        # provider boot loop runs later (ASGI lifespan) — a boot()-time
        # load_routes_from is too late for a package.
        mcp = self._settings().mcp
        if mcp.enabled:
            _import_tools(self.app.base_path, mcp.tools_dir, mcp.tools)
            self.load_routes_from(str(Path(__file__).parent / "routes.py"))

    def _settings(self) -> AiSettings:
        # scope to THIS app's config section (like Manager(app) does), not global
        return AiSettings.from_source(self.app.config("ai"))

    def boot(self) -> None:
        self.commands(cli)

        # Register the AI gateway as a resource so it shows up in the startup log and on
        # /health, and gets closed at shutdown.
        from .resource import AiResource

        self.app.resources.register(
            AiResource(self.app.make("ai"), critical=self._settings().critical)
        )
