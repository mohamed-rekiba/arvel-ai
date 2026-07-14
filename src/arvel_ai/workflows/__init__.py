"""Durable, long-running executions — the arvel-ai workflow surface.

Register a workflow function, start it, signal it, query its status through one
stable contract; swap the engine in config:

    from arvel_ai.workflows import workflow, Workflow

    @workflow(name="onboard")
    async def onboard(ctx, user_id: int) -> str:
        approved = await ctx.wait_signal("approved")
        return "welcome" if approved else "rejected"

    handle = await Workflow.start("onboard", 42)
    await Workflow.signal(handle.id, "approved", True)
    status = await Workflow.status(handle.id)   # running | completed | failed

Drivers (config `ai.workflows.default`):
- `queue`    — the default: runs the workflow on arvel.queue, tracks state in
               arvel.cache. Cooperative signals. No new infrastructure.
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
