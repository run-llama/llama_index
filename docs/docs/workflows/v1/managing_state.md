# Managing State

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
```

Then, simply annotate your workflow state with the state model:

```python
from llama_index.core.workflow import (
    Context,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)


class MyWorkflow(Workflow):
    @step
    async def start(self, ctx: Context[MyState], ev: StartEvent) -> StopEvent:
        # Allows for atomic state updates
        async with ctx.store.edit_state() as ctx_state:
            ctx_state["state"]["my_obj"]["name"] = "new_name"

        # Can also access fields directly if needed
        name = await ctx.store.get("my_obj.name")

        return StopEvent(result="Done!")
```

## Maintaining Context Across Runs

As you have seen, workflows have a `Context` object that can be used to maintain state across steps.

If you want to maintain state across multiple runs of a workflow, you can pass a previous context into the `.run()` method.

```python
handler = w.run()
result = await handler

# continue with next run
handler = w.run(ctx=handler.ctx)
result = await handler
```
