"""Type stub for the ``Workflow`` facade — restores static completion + type-safety on a
surface otherwise opaque to type-checkers (``Facade.__getattr__`` proxies to WorkflowManager).
Hand-maintained to mirror ``WorkflowManager``'s public methods (start/signal/status)."""

from typing import Any

from arvel.support.facades import Facade

from .contracts import WorkflowHandle, WorkflowStatus
from .drivers.fake import FakeWorkflowDriver

class Workflow(Facade):
    @classmethod
    async def start(cls, name: str, *args: Any, **kwargs: Any) -> WorkflowHandle: ...
    @classmethod
    async def signal(cls, workflow_id: str, name: str, payload: Any = None) -> None: ...
    @classmethod
    async def status(cls, workflow_id: str) -> WorkflowStatus: ...
    @classmethod
    def fake(cls) -> FakeWorkflowDriver: ...
