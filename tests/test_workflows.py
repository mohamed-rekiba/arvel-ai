"""Workflow contract: the registry, the queue-backed default driver, the fake, and the
manager facade. (The Temporal driver is tested against a real Temporal server in
tests/test_workflow_temporal.py.)"""

from __future__ import annotations

import pytest

from arvel.kernel import Application

from arvel_ai.workflows import Workflow, WorkflowHandle, registry, workflow
from arvel_ai.workflows.drivers.fake import FakeWorkflowDriver
from arvel_ai.workflows.drivers.queue import QueueWorkflowDriver
from arvel_ai.workflows.manager import WorkflowManager


# ---- registry ---------------------------------------------------------------


def test_workflow_decorator_registers() -> None:
    @workflow(name="probe_wf")
    async def probe(ctx: object) -> str:
        return "done"

    try:
        assert "probe_wf" in registry.names()
        assert registry.get("probe_wf") is probe
    finally:
        registry.remove("probe_wf")


def test_registry_unknown_raises() -> None:
    with pytest.raises(KeyError):
        registry.get("nope")


# ---- queue-backed default driver -------------------------------------------


@pytest.fixture()
def wf_app() -> Application:
    """A booted app with an in-memory cache + a synchronous queue stub, so the
    queue driver runs the workflow inline for the test."""
    application = Application()

    async def run_inline(coro_fn: object, *args: object) -> None:
        # the queue driver hands us the coroutine to enqueue; run it now
        await coro_fn(*args)  # type: ignore[operator]

    application.singleton("workflow.runner", lambda c: run_inline)
    return application


async def test_queue_driver_runs_and_completes(wf_app: Application) -> None:
    @workflow(name="add_wf")
    async def add(ctx: object, a: int, b: int) -> int:
        return a + b

    driver = QueueWorkflowDriver(wf_app)
    try:
        handle = await driver.start("add_wf", (2, 3), {})
        assert isinstance(handle, WorkflowHandle)
        status = await driver.status(handle.id)
        assert status.state == "completed"
        assert status.result == 5
    finally:
        registry.remove("add_wf")


async def test_queue_driver_signal_unblocks_a_waiting_workflow(wf_app: Application) -> None:
    @workflow(name="gate_wf")
    async def gate(ctx: object) -> str:
        approved = await ctx.wait_signal("approved")  # type: ignore[attr-defined]
        return "yes" if approved else "no"

    driver = QueueWorkflowDriver(wf_app)
    try:
        # pre-deliver the signal so the inline run finds it waiting
        handle = await driver.start_deferred("gate_wf", (), {})
        await driver.signal(handle.id, "approved", True)
        await driver.resume(handle.id)
        status = await driver.status(handle.id)
        assert status.state == "completed"
        assert status.result == "yes"
    finally:
        registry.remove("gate_wf")


async def test_queue_driver_records_failure(wf_app: Application) -> None:
    @workflow(name="boom_wf")
    async def boom(ctx: object) -> None:
        raise RuntimeError("kaboom")

    driver = QueueWorkflowDriver(wf_app)
    try:
        handle = await driver.start("boom_wf", (), {})
        status = await driver.status(handle.id)
        assert status.state == "failed"
        assert "kaboom" in (status.error or "")
    finally:
        registry.remove("boom_wf")


async def test_status_is_visible_across_driver_instances_via_cache() -> None:
    # a `cache` binding makes status cross-process: a second driver instance (as if in the
    # worker process) sees what the first (web process) wrote — backed by arvel.cache.
    from arvel.cache import CacheManager

    app = Application()
    app.singleton("cache", lambda _c: CacheManager())  # default array store, shared

    @workflow(name="shared_wf")
    async def shared(ctx: object, x: int) -> int:
        return x * 2

    try:
        writer = QueueWorkflowDriver(app)
        reader = QueueWorkflowDriver(app)  # a distinct instance, sharing only the app's cache
        handle = await writer.start("shared_wf", (21,), {})
        status = await reader.status(handle.id)
        assert status.state == "completed"
        assert status.result == 42
    finally:
        registry.remove("shared_wf")


# ---- fake driver + facade ---------------------------------------------------


async def test_fake_driver_scripts_status() -> None:
    fake = FakeWorkflowDriver()
    handle = await fake.start("anything", (1,), {})
    await fake.signal(handle.id, "go", True)
    status = await fake.status(handle.id)
    assert status.state == "completed"
    assert fake.started == [("anything", (1,), {})]
    assert fake.signals == [(handle.id, "go", True)]


async def test_manager_dispatches_to_configured_driver() -> None:
    application = Application()
    application.make("config").set("ai", {"workflows": {"default": "fake"}})
    manager = WorkflowManager(application)
    handle = await manager.start("wf", 1, 2)
    assert isinstance(handle, WorkflowHandle)
    assert manager.driver().started[-1][0] == "wf"


async def test_facade_fake_is_a_driver_swap() -> None:
    fake = Workflow.fake()
    assert isinstance(fake, FakeWorkflowDriver)
    handle = await Workflow.start("wf")
    await Workflow.status(handle.id)
    assert fake.started[-1][0] == "wf"
    Workflow.clear_swapped()
