"""Queue-backed workflow driver — the default. Runs a workflow function via the
app's runner (arvel.queue in production) and tracks state in a store (arvel.cache
when the app has a `cache` binding — visible across the web and worker processes;
an in-process dict otherwise).

Honest ceiling: signals are **cooperative** — `wait_signal` reads what's already
been delivered to the store; it does not durably block a suspended execution the
way a real engine does. For long-lived, retry-across-hours, human-in-the-loop
workflows use the `temporal` driver. This default covers short multi-step jobs
and reserves the contract so upgrading is a config change.
"""

from __future__ import annotations

from typing import Any

import msgspec

from arvel.support import Str

from arvel_ai.workflows.contracts import (
    WorkflowHandle,
    WorkflowStatus,
    registry,
)


class _StateStore:
    """Status + signals for the queue driver. Backed by `arvel.cache` when the app has a
    `cache` binding (cross-process visibility), else an in-process dict (single-process
    default). Values are msgspec-encoded so a serializing cache backend round-trips them —
    which means a workflow `result` must be msgspec-encodable (JSON-native or a Struct)."""

    def __init__(self, app: Any = None) -> None:
        self._cache = app.make("cache") if (app is not None and app.bound("cache")) else None
        self._local: dict[str, bytes] = {}

    async def _put(self, key: str, value: bytes) -> None:
        if self._cache is not None:
            await self._cache.put(key, value)
        else:
            self._local[key] = value

    async def _get(self, key: str) -> bytes | None:
        if self._cache is not None:
            cached: bytes | None = await self._cache.get(key)  # CacheManager.get is untyped (Any)
            return cached
        return self._local.get(key)

    async def set_status(self, status: WorkflowStatus) -> None:
        await self._put(f"workflow:status:{status.id}", msgspec.json.encode(status))

    async def get_status(self, workflow_id: str) -> WorkflowStatus | None:
        raw = await self._get(f"workflow:status:{workflow_id}")
        return msgspec.json.decode(raw, type=WorkflowStatus) if raw else None

    async def get_signals(self, workflow_id: str) -> dict[str, Any]:
        raw = await self._get(f"workflow:signals:{workflow_id}")
        return msgspec.json.decode(raw, type=dict) if raw else {}

    async def add_signal(self, workflow_id: str, name: str, payload: Any) -> None:
        signals = await self.get_signals(workflow_id)
        signals[name] = payload
        await self._put(f"workflow:signals:{workflow_id}", msgspec.json.encode(signals))


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
        self._store = _StateStore(app)
        # deferred-start closure args are process-local by nature (resume runs in the
        # process that deferred); only status + signals need cross-process visibility.
        self._pending: dict[str, tuple[str, tuple[Any, ...], dict[str, Any]]] = {}

    def _runner(self) -> Any:
        # test/override seam; production binds a runner that enqueues a Job
        if self.app is not None and self.app.bound("workflow.runner"):
            return self.app.make("workflow.runner")
        return None

    def _new_id(self, name: str) -> WorkflowHandle:
        # a globally-unique id (uuid7, time-ordered) so ids never collide across processes
        return WorkflowHandle(id=f"wf-{name}-{Str.uuid()}", name=name)

    async def _execute(
        self, handle: WorkflowHandle, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        fn = registry.get(handle.name)
        ctx = _QueueContext(handle.id, await self._store.get_signals(handle.id))
        try:
            result = await fn(ctx, *args, **kwargs)
            await self._store.set_status(
                WorkflowStatus(id=handle.id, name=handle.name, state="completed", result=result)
            )
        except Exception as exc:
            await self._store.set_status(
                WorkflowStatus(id=handle.id, name=handle.name, state="failed", error=str(exc))
            )

    # -- contract ---------------------------------------------------------------

    async def start(
        self, name: str, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None
    ) -> WorkflowHandle:
        handle = self._new_id(name)
        await self._store.set_status(WorkflowStatus(id=handle.id, name=name, state="running"))
        runner = self._runner()
        if runner is not None:
            await runner(self._execute, handle, args, kwargs or {})
        else:  # no runner bound → run inline (the honest default for one process)
            await self._execute(handle, args, kwargs or {})
        return handle

    async def signal(self, workflow_id: str, name: str, payload: Any = None) -> None:
        await self._store.add_signal(workflow_id, name, payload)

    async def status(self, workflow_id: str) -> WorkflowStatus:
        status = await self._store.get_status(workflow_id)
        if status is None:
            raise KeyError(f"no workflow {workflow_id!r}")
        return status

    # -- deferred start (signal before run) — used when a signal must be present
    #    before the (cooperative) execution reads it ----------------------------

    async def start_deferred(
        self, name: str, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None
    ) -> WorkflowHandle:
        handle = self._new_id(name)
        await self._store.set_status(WorkflowStatus(id=handle.id, name=name, state="running"))
        self._pending[handle.id] = (name, args, kwargs or {})
        return handle

    async def resume(self, workflow_id: str) -> None:
        if workflow_id not in self._pending:
            raise KeyError(f"no deferred workflow {workflow_id!r} to resume")
        name, args, kwargs = self._pending.pop(workflow_id)
        await self._execute(WorkflowHandle(id=workflow_id, name=name), args, kwargs)
