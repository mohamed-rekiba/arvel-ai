"""Durable, long-running executions — the arvel-ai workflow surface.

Register a workflow function, start it, signal it, query its status through one
stable contract; swap the engine in config:

    from arvel_ai.workflows import workflow, Workflow

    @workflow(name="onboard")
    async def onboard(ctx, user_id: int) -> str:
        approved = await ctx.wait_signal("approved")   # durably suspends (temporal)
        return "welcome" if approved else "rejected"

    handle = await Workflow.start("onboard", 42)
    await Workflow.signal(handle.id, "approved", True)
    status = await Workflow.status(handle.id)   # running | completed | failed

This shape is the *contract*. Whether `signal` can arrive after `start` and
still be observed depends on the driver: the `temporal` driver durably suspends
the execution at `wait_signal` (real human-in-the-loop). The `queue` default is
cooperative — see its ceiling below.

Drivers (config `ai.workflows.default`):
- `queue`    — the default: runs the workflow on the app runner (arvel.queue in
               production). Signals are COOPERATIVE — `wait_signal` reads what's
               already delivered; it does not durably suspend a running
               execution. Right for short, mostly-linear jobs; for a real
               post-start signal pause use `temporal`.
- `temporal` — real durable execution (retries across hours, human-in-the-loop)
               via the Temporal SDK: `uv add 'arvel-ai[temporal]'`.
- `fake`     — the test double.
"""

from .contracts import (
    Workflow,
    WorkflowContext,
    WorkflowDriver,
    WorkflowHandle,
    WorkflowState,
    WorkflowStatus,
    registry,
    workflow,
)

__all__ = [
    "Workflow",
    "WorkflowContext",
    "WorkflowDriver",
    "WorkflowHandle",
    "WorkflowState",
    "WorkflowStatus",
    "registry",
    "workflow",
]
