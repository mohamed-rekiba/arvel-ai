# Workflows

Some AI work outlives a single request — a multi-step agent run, a human-in-the-loop approval,
retries across hours. arvel's fire-and-forget [queue](gateway.md) isn't the right shape for that:
you need **durable execution** that survives restarts and can suspend waiting for a signal.
`arvel-ai` gives you one stable workflow API with a swappable engine.

```python
from arvel_ai.workflows import workflow, Workflow
```

## Define a workflow

Decorate an async function. Its first parameter is the workflow **context**; the rest are your
arguments:

```python
@workflow(name="onboard")
async def onboard(ctx, user_id: int) -> str:
    draft = await generate_welcome(user_id)
    approved = await ctx.wait_signal("approved")   # suspend until a signal arrives
    if not approved:
        return "rejected"
    await publish(user_id, draft)
    return "welcome"
```

## Drive it

```python
handle = await Workflow.start("onboard", 42)        # WorkflowHandle(id=…, name=…)

status = await Workflow.status(handle.id)           # WorkflowStatus
status.state        # running | completed | failed
status.result       # the return value once completed
status.error        # the error string if it failed

await Workflow.signal(handle.id, "approved", True)  # deliver a signal by name
```

`Workflow.start(name, *args, **kwargs)` passes `*args`/`**kwargs` straight to your function (after
`ctx`). The contract is engine-neutral — arvel-owned `WorkflowHandle` / `WorkflowStatus`, never an
engine type — so swapping engines is a config change, not a rewrite.

## Drivers

Set `ai.workflows.default`:

| Driver | Use when | Needs |
|---|---|---|
| `queue` (default) | short multi-step jobs; no new infrastructure | just arvel |
| `temporal` | long-lived, retry-across-hours, human-in-the-loop | `uv add 'arvel-ai[temporal]'` + a Temporal server |
| `fake` | tests (`Workflow.fake()`) | — |

**The `queue` default has an honest ceiling:** it runs the workflow on `arvel.queue` and tracks
state in `arvel.cache` (so status and signals are visible across your web and worker processes), but
signals are **cooperative** — `ctx.wait_signal` reads what's already been delivered; it does not
durably suspend a paused execution the way a real engine does. It's right for short, mostly-linear
jobs, and it reserves the contract so upgrading to Temporal is a config edit, not a rewrite.

## Temporal (real durable execution)

```python
# config/ai.py
ai = {
    "workflows": {
        "default": "temporal",
        "drivers": {"temporal": {
            "target": "temporal.internal:7233",
            "namespace": "default",
            "task_queue": "myapp",
        }},
    },
}
```

`arvel-ai` is the **client** side (start / signal / status). Running the workflows needs a Temporal
**worker** process that registers the same behaviour against Temporal — that's app deployment (see
the Temporal Python SDK's worker docs). The integration test in the package
(`test_workflow_temporal.py`) drives a full start → signal → durable-suspend → complete against a
real Temporal server, spun up on demand with testcontainers:

```bash
AI_INTEGRATION=1 uv run --extra temporal pytest tests/test_workflow_temporal.py
```

## Testing

`Workflow.fake()` swaps in a `FakeWorkflowDriver` that records `start`/`signal` calls and returns a
canned status — no queue, no Temporal. `Workflow.clear_swapped()` restores the real driver.

```python
from arvel_ai.workflows import Workflow

async def test_onboard_flow():
    fake = Workflow.fake()
    handle = await Workflow.start("onboard", 42)
    await Workflow.signal(handle.id, "approved", True)
    assert fake.started[-1] == ("onboard", (42,), {})
    assert fake.signals[-1] == (handle.id, "approved", True)
    Workflow.clear_swapped()
```

The fake reports `state="completed"` by default; construct it with a different state/result to
exercise the failure path.

## Common mistakes & gotchas

- **`temporal` driver needs the extra** — `uv add 'arvel-ai[temporal]'`; a missing engine tells you
  exactly that.
- **The queue driver's signals are cooperative** — don't build a long human-in-the-loop pause on it;
  use Temporal for that.
- **Register the workflow before starting it** — the `@workflow(name=...)` module must have been
  imported, or `start` raises `KeyError`.

## See also

- [The Gateway](gateway.md) · [MCP Server](mcp.md) — the rest of the AI package.
- [Configuration](configuration.md) — every `config("ai.workflows")` key and default.
