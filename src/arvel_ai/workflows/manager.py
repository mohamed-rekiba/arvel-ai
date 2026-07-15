"""WorkflowManager — config-selected workflow driver (house Manager pattern).

`Workflow.start(name, *args)` lands here; the manager builds no request objects,
just dispatches to the configured engine.
"""

from __future__ import annotations

from typing import Any, cast

from arvel.support.manager import Manager

from ..settings import WorkflowSettings
from .contracts import WorkflowDriver, WorkflowHandle, WorkflowStatus


class WorkflowManager(Manager):
    extra_package = "arvel-ai"

    def _driver(self) -> WorkflowDriver:
        return cast(WorkflowDriver, self.driver())

    def workflow_settings(self) -> WorkflowSettings:
        # base Manager._settings(cls) reads THIS manager's app config section
        from ..settings import AiSettings

        return self._settings(AiSettings).workflows

    def default_driver(self) -> str:
        return self.workflow_settings().default

    def create_queue_driver(self) -> Any:
        from .drivers.queue import QueueWorkflowDriver

        return QueueWorkflowDriver(self.app)

    def create_fake_driver(self) -> Any:
        from .drivers.fake import FakeWorkflowDriver

        return FakeWorkflowDriver()

    def create_temporal_driver(self) -> Any:
        from .drivers.temporal import TemporalWorkflowDriver

        temporal = self.workflow_settings().drivers.temporal
        return TemporalWorkflowDriver(
            target=temporal.target, namespace=temporal.namespace, task_queue=temporal.task_queue
        )

    # -- caller-facing sugar ----------------------------------------------------

    async def start(self, name: str, *args: Any, **kwargs: Any) -> WorkflowHandle:
        return await self._driver().start(name, args, kwargs)

    async def signal(self, workflow_id: str, name: str, payload: Any = None) -> None:
        await self._driver().signal(workflow_id, name, payload)

    async def status(self, workflow_id: str) -> WorkflowStatus:
        return await self._driver().status(workflow_id)
