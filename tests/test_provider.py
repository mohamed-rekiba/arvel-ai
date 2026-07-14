"""The skeleton's own acceptance tests — replace as you build.

They double as the pattern for testing YOUR package: bindings resolve, config
merges, driver swap is config-only, the facade fakes like Mail.fake().
"""

from __future__ import annotations

from arvel.kernel import Application

from arvel_ai import Ai
from arvel_ai.commands import cli
from arvel_ai.drivers import FakeDriver


def test_manager_binds_and_config_merges(app: Application) -> None:
    manager = app.make("ai")
    assert app.make("config").get("ai.default") == "memory"
    assert manager.send("ping") == "sent: ping"


def test_named_driver_dispatch(app: Application) -> None:
    manager = app.make("ai")
    assert manager.driver("fake").send("ping") == "ok"


def test_facade_fake_records_calls(app: Application) -> None:
    fake = Ai.fake()
    assert isinstance(fake, FakeDriver)
    Ai.send("hello")
    fake.assert_sent("hello")


def test_commands_are_registered(app: Application) -> None:
    assert cli in app.registry("console.commands", list)
