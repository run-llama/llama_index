# Drawing a Workflow

Workflows can be visualized, using the power of type annotations in your step definitions. You can either draw all possible paths through the workflow, or the most recent execution, to help with debugging.

First install:

```bash
pip install llama-index-utils-workflow
```

Then import and use:

```python
from llama_index.utils.workflow import (
    draw_all_possible_flows,
    draw_most_recent_execution,
)

# Draw all
draw_all_possible_flows(JokeFlow, filename="joke_flow_all.html")

# Draw an execution
w = JokeFlow()
await w.run(topic="Pirates")
draw_most_recent_execution(w, filename="joke_flow_recent.html")
```

<div id="working-with-global-context-state"></div>
## Working with Global Context/State

Optionally, you can choose to use global context between steps. For example, maybe multiple steps access the original `query` input from the user. You can store this in global context so that every step has access.

```python
from llama_index.core.workflow import Context


@step
async def query(self, ctx: Context, ev: MyEvent) -> StopEvent:
    # retrieve from context
    query = await ctx.store.get("query")

    # do something with context and event
    val = ...
    result = ...

    # store in context
    await ctx.store.set("key", val)

    return StopEvent(result=result)
```
