# Resources

Resources are external dependencies such as memory, LLMs, query engines or chat history instances that will be injected
into workflow steps at runtime.

Resources are a powerful way of binding workflow steps to Python objects that we otherwise would need to create by hand
every time. For performance reasons, by default resources are cached for a workflow, meaning the same resource instance
is passed to every step where it's injected. It's important to master this concept because cached and non-cached
resources can lead to unexpected behaviour, let's see it in detail.

## Resources are cached by default

First of all, to use resources within our code, we need to import `Resource` from the `resource` submodule:

```python
from llama_index.core.workflow.resource import Resource
from llama_index.core.workflow import (
    Event,
    step,
    StartEvent,
    StopEvent,
    Workflow,
)
```

`Resource` wraps a function or callable that must return an object of the same type as the one in the resource
definition, let's see an example:

```python
from typing import Annotated
from llama_index.core.memory import Memory


def get_memory(*args, **kwargs) -> Memory:
    return Memory.from_defaults("user_id_123", token_limit=60000)


resource = Annotated[Memory, Resource(get_memory)]
```

In the example above, `Annotated[Memory, Resource(get_memory)` defines a resource of type `Memory` that will be provided
at runtime by the `get_memory()` function. A resource defined like this can be injected into a step by passing it as
a method parameter:

```python
import random

from typing import Union
from llama_index.core.llms import ChatMessage

RANDOM_MESSAGES = [
    "Hello World!",
    "Python is awesome!",
    "Resources are great!",
]


class CustomStartEvent(StartEvent):
    message: str


class SecondEvent(Event):
    message: str


class ThirdEvent(Event):
    message: str


class WorkflowWithMemory(Workflow):
    @step
    async def first_step(
        self,
        ev: CustomStartEvent,
        memory: Annotated[Memory, Resource(get_memory)],
    ) -> SecondEvent:
        await memory.aput(
            ChatMessage.from_str(
                role="user", content="First step: " + ev.message
            )
        )
        return SecondEvent(message=RANDOM_MESSAGES[random.randint(0, 2)])

    @step
    async def second_step(
        self, ev: SecondEvent, memory: Annotated[Memory, Resource(get_memory)]
    ) -> Union[ThirdEvent, StopEvent]:
        await memory.aput(
            ChatMessage(role="assistant", content="Second step: " + ev.message)
        )
        if random.randint(0, 1) == 0:
            return ThirdEvent(message=RANDOM_MESSAGES[random.randint(0, 2)])
        else:
            messages = await memory.aget_all()
            return StopEvent(result=messages)

    @step
    async def third_step(
        self, ev: ThirdEvent, memory: Annotated[Memory, Resource(get_memory)]
    ) -> StopEvent:
        await memory.aput(
            ChatMessage(role="user", content="Third step: " + ev.message)
        )
        messages = await memory.aget_all()
        return StopEvent(result=messages)
```

As you can see, each step has access to the `memory` resource and can write to it. It's important to note that
`get_memory()` will be called only once, and the same memory instance will be injected into the different steps. We can
see this is the case by running the workflow:

```python
wf = WorkflowWithMemory(disable_validation=True)


async def main():
    messages = await wf.run(
        start_event=CustomStartEvent(message="Happy birthday!")
    )
    for m in messages:
        print(m.blocks[0].text)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```

A potential result for this might be:

```text
First step: Happy birthday!
Second step: Python is awesome!
Third step: Hello World!
```

This shows that each step added its message to a global memory, which is exactly what we were expecting!

Note that resources are preserved across steps of the same workflow instance, but not across different workflows. If we
were to run two `WorkflowWithMemory` instances, `get_memory` would be called one time for each workflow and as a result
their memories would be separate and independent:

```python
wf1 = WorkflowWithMemory(disable_validation=True)
wf2 = WorkflowWithMemory(disable_validation=True)


async def main():
    messages1 = await wf1.run(
        start_event=CustomStartEvent(message="Happy birthday!")
    )
    messages2 = await wf1.run(
        start_event=CustomStartEvent(message="Happy New Year!")
    )
    for m in messages1:
        print(m.blocks[0].text)
    print("===================")
    for m in messages2:
        print(m.blocks[0].text)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```

This is a possible output:

```text
First step: Happy birthday!
Second step: Resources are great!
===================
First step: Happy New Year!
Second step: Python is awesome!
```

## Disable resource caching

If we pass `cache=False` to `Resource` when defining a resource, the wrapped function is called every time the resource
is injected into a step. This behaviour can be desirable at times, let's see a simple example using a custom
`Counter` class:

```python
from pydantic import BaseModel, Field


class Counter(BaseModel):
    counter: int = Field(description="A simple counter", default=0)

    async def increment(self) -> None:
        self.counter += 1


def get_counter() -> Counter:
    return Counter()


class SecondEvent(Event):
    count: int


class WorkflowWithCounter(Workflow):
    @step
    async def first_step(
        self,
        ev: StartEvent,
        counter: Annotated[Counter, Resource(get_counter, cache=False)],
    ) -> SecondEvent:
        await counter.increment()
        return SecondEvent(count=counter.counter)

    @step
    async def second_step(
        self,
        ev: SecondEvent,
        counter: Annotated[Counter, Resource(get_counter, cache=False)],
    ) -> StopEvent:
        print("Counter at first step: ", ev.count)
        await counter.increment()
        print("Counter at second step: ", counter.counter)
        return StopEvent(result="End of Workflow")
```

If we now run this workflow, we will get out:

```text
Counter at first step:  1
Counter at second step:  1
```

## A note about stateful and stateless resources

As we have seen, cached resources are expected to be **stateful**, meaning that they can maintain their state across
different workflow runs and different steps, unless otherwise specified. But this doesn't mean we can consider a
resource **stateless** only because we disable caching. Let's see an example:

```python
global_mem = Memory.from_defaults("global_id", token_limit=60000)


def get_memory(*args, **kwargs) -> Memory:
    return global_mem
```

If we disable caching with `Annotated[Memory, Resource(get_memory, cache=False)]`, the function `get_memory` is going
to be called multiple times but the resource instance will be always the same. Such a resource should be considered
stateful not regarding its caching behaviour.

Now that we've mastered resources, let's take a look at [observability and debugging](./observability.md) in workflows.
