"""Provider wiring: bindings resolve, config merges, commands register."""

from __future__ import annotations

from arvel.kernel import Application

from arvel_ai.commands import cli
from arvel_ai.manager import AiManager


def test_manager_binds_and_config_merges(app: Application) -> None:
    manager = app.make("ai")
    assert isinstance(manager, AiManager)
    assert app.make("config").get("ai.default") == "litellm"
    assert app.make("config").get("ai.drivers.fake") == {}


def test_commands_are_registered(app: Application) -> None:
    assert cli in app.registry("console.commands", list)


def test_missing_extra_hint_names_this_distribution(app: Application) -> None:
    app.make("config").get("ai")["default"] = "nonexistent"
    manager = app.make("ai")
    try:
        manager.driver()
    except Exception as exc:  # MissingExtraError
        assert "arvel-ai[nonexistent]" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected MissingExtraError")
