"""Queue-backed workflow driver — the default. Runs a workflow function via the
app's runner (arvel.queue in production) and tracks state in a store (arvel.cache
in production; an in-process dict otherwise).

Honest ceiling: signals are **cooperative** — `wait_signal` reads what's already
been delivered to the store; it does not durably block a suspended execution the
way a real engine does. For long-lived, retry-across-hours, human-in-the-loop
workflows use the `temporal` driver. This default covers short multi-step jobs
and reserves the contract so upgrading is a config change.
"""

from __future__ import annotations

from typing import Any

from arvel_ai.workflows.contracts import (
    WorkflowHandle,
    WorkflowStatus,
    registry,
)


class _QueueContext:
    """The context a queue-run workflow receives. `wait_signal` reads a signal
    already delivered to this workflow's slot (cooperative)."""

    def __init__(self, workflow_id: str, signals: dict[str, Any]) -> None:
        self.id = workflow_id
        self._signals = signals

    async def wait_signal(self, name: str, default: Any = None) -> Any:
        return self._signals.get(name, default)


class QueueWorkflowDriver:
    def __init__(self, app: Any = None) -> None:
        self.app = app
        # ponytail: in-process store — swap for arvel.cache (put/get) to make
        # status + signals visible across the web and worker processes
        self._status: dict[str, WorkflowStatus] = {}
        self._signals: dict[str, dict[str, Any]] = {}
        self._pending: dict[str, tuple[str, tuple[Any, ...], dict[str, Any]]] = {}
        self._counter = 0

    def _runner(self) -> Any:
        # test/override seam; production binds a runner that enqueues a Job
        if self.app is not None and self.app.bound("workflow.runner"):
            return self.app.make("workflow.runner")
        return None

    def _new_id(self, name: str) -> WorkflowHandle:
        self._counter += 1
        wid = f"wf-{name}-{self._counter}"
        self._signals.setdefault(wid, {})
        return WorkflowHandle(id=wid, name=name)

    async def _execute(self, handle: WorkflowHandle, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        fn = registry.get(handle.name)
        ctx = _QueueContext(handle.id, self._signals.get(handle.id, {}))
        try:
            result = await fn(ctx, *args, **kwargs)
            self._status[handle.id] = WorkflowStatus(
                id=handle.id, name=handle.name, state="completed", result=result
            )
        except Exception as exc:
            self._status[handle.id] = WorkflowStatus(
                id=handle.id, name=handle.name, state="failed", error=str(exc)
            )

    # -- contract ---------------------------------------------------------------

    async def start(
        self, name: str, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None
    ) -> WorkflowHandle:
        handle = self._new_id(name)
        self._status[handle.id] = WorkflowStatus(id=handle.id, name=name, state="running")
        runner = self._runner()
        if runner is not None:
            await runner(self._execute, handle, args, kwargs or {})
        else:  # no runner bound → run inline (the honest default for one process)
            await self._execute(handle, args, kwargs or {})
        return handle

    async def signal(self, workflow_id: str, name: str, payload: Any = None) -> None:
        self._signals.setdefault(workflow_id, {})[name] = payload

    async def status(self, workflow_id: str) -> WorkflowStatus:
        if workflow_id not in self._status:
            raise KeyError(f"no workflow {workflow_id!r}")
        return self._status[workflow_id]

    # -- deferred start (signal before run) — used when a signal must be present
    #    before the (cooperative) execution reads it ----------------------------

    async def start_deferred(
        self, name: str, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None
    ) -> WorkflowHandle:
        handle = self._new_id(name)
        self._status[handle.id] = WorkflowStatus(id=handle.id, name=name, state="running")
        self._pending[handle.id] = (name, args, kwargs or {})
        return handle

    async def resume(self, workflow_id: str) -> None:
        if workflow_id not in self._pending:
            raise KeyError(f"no deferred workflow {workflow_id!r} to resume")
        name, args, kwargs = self._pending.pop(workflow_id)
        await self._execute(WorkflowHandle(id=workflow_id, name=name), args, kwargs)
