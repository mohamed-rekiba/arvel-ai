"""AiResource — the AI gateway as a lifecycle-managed, health-checkable resource (DR-0039).

Registered by AiServiceProvider so a booting app reports the AI backend in its
resource-startup log and /health, and drains driver-held clients (e.g. the
openai_compatible httpx client) at shutdown. Non-critical: an AI outage degrades
rather than aborts boot — most apps can still serve without AI.
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
        """Report the configured driver. A `health()` on the driver (DR-0039
        ManagedLifecycle) is used when present; otherwise resolving the driver
        is itself the reachability signal — a misconfigured/missing extra
        surfaces here as a degraded resource instead of a first-call 500."""
        driver_name = self._driver_name()
        try:
            driver = self._manager.driver()
        except Exception as exc:  # MissingExtraError / bad config
            return HealthResult(HealthStatus.FAILED, detail=f"{driver_name}: {exc}")
        health = getattr(driver, "health", None)
        if callable(health):
            result = health()
            if hasattr(result, "__await__"):
                result = await result
            return result  # type: ignore[no-any-return]
        return HealthResult(HealthStatus.OK, detail=f"{driver_name} (configured)")

    async def disconnect(self) -> None:
        """Drain any client the resolved driver holds (DR-0039 teardown)."""
        try:
            driver = self._manager.driver()
        except Exception:
            return
        close = getattr(driver, "aclose", None) or getattr(driver, "close", None)
        if callable(close):
            result = close()
            if hasattr(result, "__await__"):
                await result
