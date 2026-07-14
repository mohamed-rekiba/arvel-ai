"""`AI.chat(...)` — static-looking access to the container-resolved AiManager.

`AI.fake()` swaps in the FakeAiDriver for tests, same verb as `Mail.fake()`:
script `fake.replies`, assert with `fake.assert_chatted(...)`.
(IDE stubs: hand-write a .pyi or wait for `arvel stubs:generate`.)
"""

from __future__ import annotations

from arvel.support.facades import Facade


class AI(Facade):
    @classmethod
    def accessor(cls) -> str:
        return "ai"

    @classmethod
    def fake_class(cls) -> type:
        from .drivers.fake import FakeAiDriver

        return FakeAiDriver
