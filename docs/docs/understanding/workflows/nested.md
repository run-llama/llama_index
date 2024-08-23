# Nested workflows

Another way to extend workflows is to nest additional workflows. It's possible to create explicit slots in existing flows where you can supply an entire additional workflow. For example, let's say we had a query that used an LLM to reflect on the quality of that query. The author might expect that you would want to modify the reflection step, and leave a slot for you to do that.

Here's our base workflow:

```python
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context,
)
from llama_index.utils.workflow import draw_all_possible_flows


class Step2Event(Event):
    query: str


class MainWorkflow(Workflow):
    @step
    async def start(
        self, ctx: Context, ev: StartEvent, reflection_workflow: Workflow
    ) -> Step2Event:
        print("Need to run reflection")
        res = await reflection_workflow.run(query=ev.query)

        return Step2Event(query=res)

    @step
    async def step_two(self, ctx: Context, ev: Step2Event) -> StopEvent:
        print("Query is ", ev.query)
        # do something with the query here
        return StopEvent(result=ev.query)
```

This workflow by itself will not run; it needs a valid workflow for the reflection step. Let's create one:

```python
class ReflectionFlow(Workflow):
    @step
    async def sub_start(self, ctx: Context, ev: StartEvent) -> StopEvent:
        print("Doing custom reflection")
        return StopEvent(result="Improved query")
```

Now we can run the main workflow by supplying this custom reflection nested flow using the `add_workflows` method, to which we pass an instance of the `ReflectionFlow` class:

```python
w = MainWorkflow(timeout=10, verbose=False)
w.add_workflows(reflection_workflow=ReflectionFlow())
result = await w.run(query="Initial query")
print(result)
```

Note that because the nested flow is a totally different workflow rather than a step, `draw_all_possible_flows` will only draw the flow of `MainWorkflow`.

Finally, let's take a look at [observability and debugging](observability.md) in workflows.
