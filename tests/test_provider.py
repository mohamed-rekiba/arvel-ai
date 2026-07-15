"""Provider wiring: bindings resolve, typed settings default, commands register."""

from __future__ import annotations

from arvel.kernel import Application

from arvel_ai.commands import cli
from arvel_ai.manager import AiManager
from arvel_ai.settings import AiSettings


def test_manager_binds_and_settings_default(app: Application) -> None:
    manager = app.make("ai")
    assert isinstance(manager, AiManager)
    # typed AiSettings supplies defaults (no merge_config_from / DEFAULTS dict)
    assert manager.settings().default == "litellm"
    assert AiSettings().default == "litellm"
    assert manager.default_driver() == "litellm"


def test_commands_are_registered(app: Application) -> None:
    assert cli in app.registry("console.commands", list)


def test_missing_extra_hint_names_this_distribution(app: Application) -> None:
    app.make("config").set("ai.default", "nonexistent")
    manager = app.make("ai")
    try:
        manager.driver()
    except Exception as exc:  # MissingExtraError
        assert "arvel-ai[nonexistent]" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected MissingExtraError")
