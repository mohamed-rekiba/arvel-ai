"""Driver dispatch for ai — config-selected backend, ecosystem-extensible.

The house Manager pattern: `default_driver()` reads config, `create_<x>_driver`
methods build backends, `.extend("x", factory)` is the seam for custom drivers,
and unknown attribute access forwards to the default driver.
"""

from __future__ import annotations

from typing import Any

from arvel.support.manager import Manager


class AiManager(Manager):
    # MissingExtraError hints name THIS distribution, not arvel core.
    extra_package = "arvel-ai"

    def default_driver(self) -> str:
        if self.app is not None:
            return str(self.app.make("config").get("ai.default", "memory"))
        return "memory"

    def create_memory_driver(self) -> Any:
        from .drivers import MemoryDriver

        return MemoryDriver()

    def create_fake_driver(self) -> Any:
        from .drivers import FakeDriver

        return FakeDriver()
