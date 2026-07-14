"""Temporal driver against a REAL Temporal server (DR-0043 + constraint 3:
real service in Docker, never a mock).

    docker compose -f docker-compose.test.yml up -d
    AI_TEMPORAL_TARGET=localhost:7233 uv run pytest tests/test_workflow_temporal.py -q

Drives a real workflow through the arvel-owned WorkflowDriver: start -> signal ->
durable-suspend-then-resume -> complete, with an in-process Temporal worker.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from arvel_ai.workflows.contracts import WorkflowHandle
from arvel_ai.workflows.drivers.temporal import TemporalWorkflowDriver

TARGET = os.environ.get("AI_TEMPORAL_TARGET")

pytestmark = pytest.mark.skipif(
    not TARGET, reason="AI_TEMPORAL_TARGET not set (start docker-compose.test.yml)"
)

# Temporal requires workflow classes to be module-level (globally name-referenceable).
if TARGET:
    from temporalio import workflow as t_workflow

    @t_workflow.defn(name="gate_wf")
    class GateWorkflow:
        def __init__(self) -> None:
            self._approved: bool | None = None

        @t_workflow.run
        async def run(self) -> str:
            await t_workflow.wait_condition(lambda: self._approved is not None)
            return "yes" if self._approved else "no"

        @t_workflow.signal
        def approved(self, value: bool) -> None:  # noqa: FBT001
            self._approved = value


@pytest.fixture()
async def task_queue():  # type: ignore[no-untyped-def]
    """Run a real Temporal worker against the server for the test's duration."""
    from temporalio.client import Client
    from temporalio.worker import Worker

    queue = f"arvel-ai-test-{uuid.uuid4().hex[:8]}"
    client = await Client.connect(TARGET, namespace="default")
    worker = Worker(client, task_queue=queue, workflows=[GateWorkflow])
    task = asyncio.create_task(worker.run())
    try:
        yield queue
    finally:
        task.cancel()


async def test_real_workflow_start_signal_complete(task_queue: str) -> None:
    driver = TemporalWorkflowDriver(target=TARGET, task_queue=task_queue)

    handle = await driver.start("gate_wf", (), {})
    assert isinstance(handle, WorkflowHandle)

    # durably suspended on wait_condition until the signal lands
    status = await driver.status(handle.id)
    assert status.state == "running"

    await driver.signal(handle.id, "approved", True)

    for _ in range(50):
        status = await driver.status(handle.id)
        if status.state == "completed":
            break
        await asyncio.sleep(0.2)
    assert status.state == "completed"
    assert status.result == "yes"
