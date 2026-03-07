---
title: Nested / sub-workflows
---

# Nested and sub-workflows

The previous way to nest workflows via `add_workflows` has been **removed** and must not be used. This page describes the current way to compose workflows by creating and running sub-workflows.

## Current approach

You can create a sub-workflow in either of these ways:

1. **Inside a step** – Instantiate a child `Workflow` and call `run()` or `arun()` from within the step.
2. **In the workflow class constructor** – Store a child workflow as an instance attribute (e.g. `self.child_workflow`) and invoke it from your steps as needed.

Both patterns give you full control over when the sub-workflow runs and how you pass data in and out.

---

## Option A: Sub-workflow inside a step

Create and run the child workflow inside a step. This is useful when the sub-workflow is only needed in that step or when you want to pass step-specific inputs.

```python
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent

class ChildWorkflow(Workflow):
    @step
    async def do_work(self, ev: StartEvent) -> StopEvent:
        # Child logic here
        return StopEvent(result=ev.query.upper())

class ParentWorkflow(Workflow):
    @step
    async def run_child(self, ev: StartEvent) -> StopEvent:
        child = ChildWorkflow()
        result = await child.run(query=ev.query)
        return StopEvent(result=result)
```

---

## Option B: Sub-workflow in the constructor

Create the child workflow in `__init__` and store it on the instance. Use it from any step that needs it. This is useful when the same sub-workflow is reused across steps or when you want to configure it once at construction time.

```python
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent

class ChildWorkflow(Workflow):
    @step
    async def do_work(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result=ev.query.upper())

class ParentWorkflow(Workflow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.child_workflow = ChildWorkflow()

    @step
    async def run_child(self, ev: StartEvent) -> StopEvent:
        result = await self.child_workflow.run(query=ev.query)
        return StopEvent(result=result)
```

---

## Use case: Human-in-the-loop

You can implement human-in-the-loop by using a **separate workflow** for gathering human input and running it as a sub-workflow (from a step or from the constructor). For a full example, see the [Human in the loop story crafting](https://github.com/run-llama/llama_index/blob/main/docs/examples/workflow/human_in_the_loop_story_crafting.ipynb) notebook. When deploying with [llama-deploy](https://github.com/run-llama/llama_deploy), that human-input workflow can be deployed as its own service.

---

## API reference

For the full `Workflow` API (events, steps, context), see the [Workflow API reference](https://docs.llamaindex.ai/en/stable/api_reference/workflow/workflow/).
