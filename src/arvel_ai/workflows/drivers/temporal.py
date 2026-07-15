"""Temporal workflow driver — real durable execution (retries across hours,
human-in-the-loop, signals that durably suspend an execution).

The temporalio SDK is confined to this module (import-linter contract),
lazy-imported, and installed via `uv add 'arvel-ai[temporal]'`. No temporalio
type crosses the WorkflowDriver boundary — arvel-owned handles/status only.

Running workflows needs a Temporal worker process; this driver is the client
side (start / signal / status). The worker wiring (registering the same
`@workflow` functions against a Temporal worker) is app deployment, documented
in packages/ai-workflows.md.
"""

from __future__ import annotations

from typing import Any

from arvel.support.manager import MissingExtraError

from arvel_ai.workflows.contracts import WorkflowHandle, WorkflowState, WorkflowStatus


class TemporalWorkflowDriver:
    def __init__(
        self,
        target: str = "localhost:7233",
        namespace: str = "default",
        task_queue: str = "arvel-ai",
    ) -> None:
        self.target = target
        self.namespace = namespace
        self.task_queue = task_queue
        self._client: Any = None

    async def _connect(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from temporalio.client import Client
        except ImportError as exc:
            raise MissingExtraError("temporal", package="arvel-ai") from exc
        self._client = await Client.connect(self.target, namespace=self.namespace)
        return self._client

    async def start(
        self, name: str, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None
    ) -> WorkflowHandle:
        client = await self._connect()

        import uuid

        workflow_id = f"{name}-{uuid.uuid7().hex[:12]}"
        # positional args map to Temporal's arg list; the registered workflow's
        # run signature is (ctx-less) — the app's worker adapts @workflow fns
        await client.start_workflow(
            name,
            *args,
            id=workflow_id,
            task_queue=self.task_queue,
        )
        return WorkflowHandle(id=workflow_id, name=name)

    async def signal(self, workflow_id: str, name: str, payload: Any = None) -> None:
        client = await self._connect()
        handle = client.get_workflow_handle(workflow_id)
        await handle.signal(name, payload)

    async def status(self, workflow_id: str) -> WorkflowStatus:
        client = await self._connect()
        handle = client.get_workflow_handle(workflow_id)
        description = await handle.describe()
        state = _map_status(str(description.status.name) if description.status else "")
        result: Any = None
        error: str | None = None
        if state == "completed":
            try:
                result = await handle.result()
            except Exception as exc:  # completed-but-unfetchable
                error = str(exc)
        elif state == "failed":
            try:
                await handle.result()
            except Exception as exc:
                error = str(exc)
        return WorkflowStatus(
            id=workflow_id, name=description.workflow_type or "", state=state, result=result, error=error
        )


def _map_status(temporal_status: str) -> WorkflowState:
    # temporalio WorkflowExecutionStatus names → the arvel taxonomy
    upper = temporal_status.upper()
    if "COMPLETED" in upper:
        return "completed"
    if any(term in upper for term in ("FAILED", "TERMINATED", "CANCELED", "TIMED_OUT")):
        return "failed"
    return "running"
