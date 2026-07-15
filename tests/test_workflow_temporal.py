"""Temporal driver against a real Temporal server in Docker — never a mock.

testcontainers spins the server up for us, so there's no manual `docker compose`:

    AI_INTEGRATION=1 uv run --extra temporal pytest tests/test_workflow_temporal.py -q

Drives a real workflow through the arvel-owned WorkflowDriver: start -> signal ->
durable-suspend-then-resume -> complete, with an in-process Temporal worker.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Iterator

import pytest

from arvel_ai.workflows.contracts import WorkflowHandle
from arvel_ai.workflows.drivers.temporal import TemporalWorkflowDriver

INTEGRATION = os.environ.get("AI_INTEGRATION") == "1"

pytestmark = pytest.mark.skipif(
    not INTEGRATION, reason="AI_INTEGRATION != 1 — real-service tier (needs Docker)"
)

# Temporal requires workflow classes to be module-level (globally name-referenceable),
# and temporalio only ships with the `temporal` extra — so import it under the flag.
if INTEGRATION:
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


@pytest.fixture(scope="module")
def temporal_target() -> Iterator[str]:
    """Start a real Temporal dev server in a container; yield its host:port."""
    from testcontainers.core.container import DockerContainer

    container = (
        DockerContainer("temporalio/temporal:1.5.1")
        .with_command("server start-dev --ip 0.0.0.0 --namespace default")
        .with_exposed_ports(7233)
    )
    container.start()
    try:
        host = container.get_container_host_ip()
        yield f"{host}:{container.get_exposed_port(7233)}"
    finally:
        container.stop()


@pytest.fixture()
async def task_queue(temporal_target: str) -> AsyncIterator[tuple[str, str]]:
    """Run a real Temporal worker against the server for the test's duration. Retries the
    initial connect while the freshly-started server finishes coming up."""
    from temporalio.client import Client
    from temporalio.worker import Worker

    client = None
    for _ in range(60):
        try:
            client = await Client.connect(temporal_target, namespace="default")
            break
        except Exception:  # noqa: BLE001 - server still booting; retry
            await asyncio.sleep(1)
    assert client is not None, "Temporal server never became reachable"

    queue = f"arvel-ai-test-{uuid.uuid4().hex[:8]}"
    worker = Worker(client, task_queue=queue, workflows=[GateWorkflow])
    task = asyncio.create_task(worker.run())
    try:
        yield queue, temporal_target
    finally:
        task.cancel()


async def test_real_workflow_start_signal_complete(task_queue: tuple[str, str]) -> None:
    queue, target = task_queue
    driver = TemporalWorkflowDriver(target=target, task_queue=queue)

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
