"""AiResource: the gateway reports its health and closes its client at shutdown, so a
booting app logs it and /health covers it — no external service needed here."""

from __future__ import annotations

from arvel.contracts import HealthStatus
from arvel.kernel import Application

from arvel_ai.provider import AiServiceProvider
from arvel_ai.resource import AiResource


async def test_resource_reports_configured_driver_healthy(app: Application) -> None:
    app.make("config").set("ai.default", "fake")
    resource = AiResource(app.make("ai"))
    result = await resource.check()
    assert result.status is HealthStatus.OK
    assert "fake" in (result.detail or "")


async def test_resource_degrades_on_missing_extra(app: Application) -> None:
    app.make("config").set("ai.default", "nonexistent")
    resource = AiResource(app.make("ai"))
    result = await resource.check()
    assert result.status is HealthStatus.FAILED
    assert "arvel-ai[nonexistent]" in (result.detail or "")


async def test_resource_disconnect_drains_a_driver_with_a_client() -> None:
    class ClosableDriver:
        supports_embeddings = False

        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    application = Application()
    provider = AiServiceProvider(application)
    provider.register()
    manager = application.make("ai")
    driver = ClosableDriver()
    manager.extend("closable", lambda _app: driver)
    application.make("config").set("ai.default", "closable")

    await AiResource(manager).disconnect()
    assert driver.closed is True


def test_provider_registers_the_resource(app: Application) -> None:
    assert any(r.name == "ai" for r in app.resources.resources)
