"""Package test harness: a booted bare Application with this provider registered.

This is the in-process equivalent of entry-point discovery — no `pip install`
needed to test the package against arvel.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from arvel.kernel import Application

from arvel_ai import AI
from arvel_ai.provider import AiServiceProvider


@pytest.fixture()
def app() -> Iterator[Application]:
    application = Application()
    provider = AiServiceProvider(application)
    provider.register()
    provider.boot()
    yield application
    AI.clear_swapped()
