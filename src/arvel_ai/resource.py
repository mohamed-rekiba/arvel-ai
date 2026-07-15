"""The AI gateway as a health-checkable resource.

AiServiceProvider registers this so a booting app reports the AI backend in the startup log
and on /health, and so any client a driver holds gets closed at shutdown. It's non-critical:
if the AI backend is down the app still starts and serves everything else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arvel.contracts import HealthResult, HealthStatus

if TYPE_CHECKING:
    from .manager import AiManager


class AiResource:
    name = "ai"

    def __init__(self, manager: AiManager, *, critical: bool = False) -> None:
        self._manager = manager
        self.critical = critical

    def _driver_name(self) -> str:
        return self._manager.default_driver()

    async def check(self) -> HealthResult:
        """Ask the driver how it's doing. If the driver has a health() we use it; if it doesn't,
        just resolving it is the signal — a missing extra or bad config shows up here as an
        unhealthy resource instead of blowing up on the first real call."""
        driver_name = self._driver_name()
        try:
            driver = self._manager.driver()
            health = getattr(driver, "health", None)
            if callable(health):
                result = health()
                if hasattr(result, "__await__"):
                    result = await result
                return result  # type: ignore[no-any-return]
        except Exception as exc:  # MissingExtraError, bad config, or a raising health()
            return HealthResult(HealthStatus.FAILED, detail=f"{driver_name}: {exc}")
        return HealthResult(HealthStatus.OK, detail=f"{driver_name} (configured)")

    async def disconnect(self) -> None:
        """Close whatever client the driver is holding, if it exposes aclose()/close()."""
        try:
            driver = self._manager.driver()
        except Exception:
            return
        close = getattr(driver, "aclose", None) or getattr(driver, "close", None)
        if callable(close):
            result = close()
            if hasattr(result, "__await__"):
                await result
