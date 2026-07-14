"""The fake workflow driver — a first-class driver (Workflow.fake()), the test
double, and this subpackage's red-green harness."""

from __future__ import annotations

from typing import Any

from arvel_ai.workflows.contracts import WorkflowHandle, WorkflowState, WorkflowStatus


class FakeWorkflowDriver:
    def __init__(self, state: WorkflowState = "completed", result: Any = None) -> None:
        self.default_state = state
        self.default_result = result
        self.started: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.signals: list[tuple[str, str, Any]] = []
        self._counter = 0

    async def start(
        self, name: str, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None
    ) -> WorkflowHandle:
        self._counter += 1
        self.started.append((name, args, kwargs or {}))
        return WorkflowHandle(id=f"fake-{self._counter}", name=name)

    async def signal(self, workflow_id: str, name: str, payload: Any = None) -> None:
        self.signals.append((workflow_id, name, payload))

    async def status(self, workflow_id: str) -> WorkflowStatus:
        name = self.started[-1][0] if self.started else "unknown"
        return WorkflowStatus(
            id=workflow_id, name=name, state=self.default_state, result=self.default_result
        )
