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
        db = await ctx.store.get("some_database", default=None)
        if db is None:
            print("Need to load data")
            return SetupEvent(query=ev.query)

        # do something with the query
        return StepTwoEvent(query=ev.query)

    @step
    async def setup(self, ctx: Context, ev: SetupEvent) -> StartEvent:
        # load data
        await ctx.store.set("some_database", [1, 2, 3])
        return StartEvent(query=ev.query)
```

Then in `step_two` we can access data directly from the context without having it passed explicitly. In gen AI applications this is useful for loading indexes and other large data operations.

```python
@step
async def step_two(self, ctx: Context, ev: StepTwoEvent) -> StopEvent:
    # do something with the data
    print("Data is ", await ctx.store.get("some_database"))

    return StopEvent(result=await ctx.store.get("some_database"))


w = StatefulFlow(timeout=10, verbose=False)
result = await w.run(query="Some query")
print(result)
```

## Handling Concurrent State Changes

When multiple agents are running in parallel, it's possible that they will try to modify the same state at the same time. This can lead to race conditions and unexpected behavior.

To avoid this, you can use a `with` statement to edit the state. This will ensure that the state is updated atomically.

```python
async with ctx.store.edit_state() as state:
    state["some_key"] = "some_value"
```

This will ensure that only one step/task can access and edit the state at a given time. If other steps/tasks need to access the state, they will wait until the current edit operation exits.

## Adding Typed State

Often, you'll have some preset shape that you want to use as the state for your workflow. The best way to do this is to use a `Pydantic` model to define the state. This way, you:

- Get type hints for your state
- Get automatic validation of your state
- (Optionally) Have full control over the serialization and deserialization of your state using [validators](https://docs.pydantic.dev/latest/concepts/validators/) and [serializers](https://docs.pydantic.dev/latest/concepts/serialization/#custom-serializers)

**NOTE:** You should use a pydantic model that has defaults for all fields. This enables the `Context` object to automatically initialize the state with the defaults.

Here's a quick example of how you can leverage workflows + pydantic to take advantage of all these features:

```python
from pydantic import BaseModel, Field, field_validator, field_serializer
from typing import Union

from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)


# This is a random object that we want to use in our state
class MyRandomObject:
    def __init__(self, name: str = "default"):
        self.name = name


# This is our state model
# NOTE: all fields must have defaults
class MyState(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    my_obj: MyRandomObject = Field(default_factory=MyRandomObject)
    some_key: str = Field(default="some_value")

    # This is optional, but can be useful if you want to control the serialization of your state!

    @field_serializer("my_obj", when_used="always")
    def serialize_my_obj(self, my_obj: MyRandomObject) -> str:
        return my_obj.name

    @field_validator("my_obj", mode="before")
    @classmethod
    def deserialize_my_obj(
        cls, v: Union[str, MyRandomObject]
    ) -> MyRandomObject:
        if isinstance(v, MyRandomObject):
            return v
        if isinstance(v, str):
            return MyRandomObject(v)

        raise ValueError(f"Invalid type for my_obj: {type(v)}")


class MyStatefulFlow(Workflow):
    @step
    async def start(self, ctx: Context[MyState], ev: StartEvent) -> StopEvent:
        # Allows for atomic state updates
        async with ctx.store.edit_state() as state:
            state.my_obj.name = "new_name"

        # Can also access fields directly if needed
        name = await ctx.store.get("my_obj.name")

        return StopEvent(result="Done!")


w = MyStatefulFlow(timeout=10, verbose=False)

ctx = Context(w)
result = await w.run(ctx=ctx)
state = await ctx.store.get_state()
print(state)
```


Up next we'll learn how to [stream events](stream.md) from an in-progress workflow.
