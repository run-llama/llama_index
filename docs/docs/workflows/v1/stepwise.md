# Stepwise Execution

Workflows have built-in utilities for stepwise execution, allowing you to control execution and debug state as things progress.

```python
# Create a workflow, same as usual
workflow = JokeFlow()
# Get the handler. Passing `stepwise=True` will block execution, waiting for manual intervention
handler = workflow.run(stepwise=True)
# Each time we call `run_step`, the workflow will advance and return all the events
# that were produced in the last step. This events need to be manually propagated
# for the workflow to keep going (we assign them to `produced_events` with the := operator).
while produced_events := await handler.run_step():
    # If we're here, it means there's at least an event we need to propagate,
    # let's do it with `send_event`
    for ev in produced_events:
        handler.ctx.send_event(ev)

# If we're here, it means the workflow execution completed, and
# we can now access the final result.
result = await handler
```

## Checkpointing Workflows

Workflow runs can also be made to create and store checkpoints upon every step completion via the `WorfklowCheckpointer` object. These checkpoints can be then be used as the starting points for future runs, which can be a helpful feature during the development (and debugging) of your Workflow.

```python
from llama_index.core.workflow import WorkflowCheckpointer

w = JokeFlow(...)
w_cptr = WorkflowCheckpointer(workflow=w)

# to checkpoint a run, use the `run` method from w_cptr
handler = w_cptr.run(topic="Pirates")
await handler

# to view the stored checkpoints of this run
w_cptr.checkpoints[handler.run_id]

# to run from one of the checkpoints, use `run_from` method
ckpt = w_cptr.checkpoints[handler.run_id][0]
handler = w_cptr.run_from(topic="Ships", checkpoint=ckpt)
await handler
```
