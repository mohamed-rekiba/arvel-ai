"""The stable workflow contract — arvel-owned, engine-neutral (mirrors the
gateway's anti-corruption stance: no temporalio type crosses this boundary)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol, runtime_checkable

import msgspec

WorkflowState = Literal["running", "completed", "failed"]


class WorkflowHandle(msgspec.Struct):
    """Returned by start(); the durable id you signal and query."""

    id: str
    name: str


class WorkflowStatus(msgspec.Struct):
    id: str
    name: str
    state: WorkflowState = "running"
    result: Any = None
    error: str | None = None


class WorkflowContext(Protocol):
    """What a workflow function receives — its handle onto the running execution.
    `wait_signal` blocks (durably, on a real engine) until the named signal
    arrives."""

    id: str

    async def wait_signal(self, name: str, default: Any = None) -> Any: ...


@runtime_checkable
class WorkflowDriver(Protocol):
    """What every workflow engine implements. Registration of the workflow
    functions is shared (the registry below); the driver only executes."""

    async def start(
        self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> WorkflowHandle: ...

    async def signal(self, workflow_id: str, name: str, payload: Any = None) -> None: ...

    async def status(self, workflow_id: str) -> WorkflowStatus: ...


# ---- registry (shared across drivers) --------------------------------------

WorkflowFn = Callable[..., Awaitable[Any]]


class WorkflowRegistry:
    def __init__(self) -> None:
        self._fns: dict[str, WorkflowFn] = {}

    def register(self, name: str | None = None) -> Callable[[WorkflowFn], WorkflowFn]:
        def decorate(fn: WorkflowFn) -> WorkflowFn:
            self._fns[name or fn.__name__] = fn
            return fn

        return decorate

    def get(self, name: str) -> WorkflowFn:
        if name not in self._fns:
            raise KeyError(f"no workflow registered as {name!r}")
        return self._fns[name]

    def names(self) -> list[str]:
        return list(self._fns)

    def remove(self, name: str) -> None:
        self._fns.pop(name, None)


registry = WorkflowRegistry()


def workflow(name: str | None = None) -> Callable[[WorkflowFn], WorkflowFn]:
    """Register a workflow function under the default registry."""
    return registry.register(name)
