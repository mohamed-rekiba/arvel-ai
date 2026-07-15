"""`Workflow.start(...)` / `.signal(...)` / `.status(...)` — static-looking access to the
container-resolved WorkflowManager. `Workflow.fake()` swaps in the FakeWorkflowDriver for tests.
(IDE stubs: facade.pyi, hand-maintained to mirror WorkflowManager.)
"""

from __future__ import annotations

from arvel.support.facades import Facade


class Workflow(Facade):
    @classmethod
    def accessor(cls) -> str:
        return "ai.workflows"

    @classmethod
    def fake_class(cls) -> type:
        from .drivers.fake import FakeWorkflowDriver

        return FakeWorkflowDriver
