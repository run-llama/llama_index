# Maintaining state

In our examples so far, we have passed data from step to step using properties of custom events. This is a powerful way to pass data around, but it has limitations. For example, if you want to pass data between steps that are not directly connected, you need to pass the data through all the steps in between. This can make your code harder to read and maintain.

To avoid this pitfall, we have a `Context` object available to every step in the workflow. To use it, declare an argument of type `Context` to your step. Here's how you do that.

We need one new import, the `Context` type:

```python
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context,
)
```

Now we define a `start` event that checks if data has been loaded into the context. If not, it returns a `SetupEvent` which triggers `setup` that loads the data and loops back to `start`.

```python
class SetupEvent(Event):
    query: str


class StepTwoEvent(Event):
    query: str


class StatefulFlow(Workflow):
    @step
    async def start(
        self, ctx: Context, ev: StartEvent
    ) -> SetupEvent | StepTwoEvent:
        db = await ctx.get("some_database", default=None)
        if db is None:
            print("Need to load data")
            return SetupEvent(query=ev.query)

        # do something with the query
        return StepTwoEvent(query=ev.query)

    @step
    async def setup(self, ctx: Context, ev: SetupEvent) -> StartEvent:
        # load data
        await ctx.set("some_database", [1, 2, 3])
        return StartEvent(query=ev.query)
```

Then in `step_two` we can access data directly from the context without having it passed explicitly. In gen AI applications this is useful for loading indexes and other large data operations.

```python
@step
async def step_two(self, ctx: Context, ev: StepTwoEvent) -> StopEvent:
    # do something with the data
    print("Data is ", await ctx.get("some_database"))

    return StopEvent(result=await ctx.get("some_database"))


w = StatefulFlow(timeout=10, verbose=False)
result = await w.run(query="Some query")
print(result)
```

Up next we'll learn how to [stream events](stream.md) from an in-progress workflow.
