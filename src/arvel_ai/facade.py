"""`Ai.send(...)` — static-looking access to the container-resolved manager.

`Ai.fake()` swaps in the FakeDriver for tests, same verb as `Mail.fake()`.
(IDE stubs: hand-write a .pyi or wait for `arvel stubs:generate`.)
"""

from __future__ import annotations

from arvel.support.facades import Facade


class Ai(Facade):
    @classmethod
    def accessor(cls) -> str:
        return "ai"

    @classmethod
    def fake_class(cls) -> type:
        from .drivers import FakeDriver

        return FakeDriver
