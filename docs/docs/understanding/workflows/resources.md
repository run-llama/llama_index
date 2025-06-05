# Resources

Resources are a component of workflows that allow us to equip our steps with external dependencies such as memory, LLMs, query engines or chat history.

Resources are a powerful way of binding components to our steps that we otherwise would need to specify by hand every time and, most importantly, resources are **stateful**, meaning that they maintain their state across different steps, unless otherwise specified.

## Using Stateful Resources

In order to use them within our code, we need to import them from the `resource` submodule:

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

The `Resource` function works as a wrapper for another function that, when executed, returns an object of a specified type. This is the usage pattern:

```python
from typing import Annotated
from llama_index.core.memory import Memory


def get_memory(*args, **kwargs) -> Memory:
    return Memory.from_defaults("user_id_123", token_limit=60000)


resource = Annotated[Memory, Resource(get_memory)]
```

When a step of our workflow will be equipped with this resource, the variable in the step to which the resource is assigned would behave as a memory component:

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

As you can see, each step has access to memory and writes to it - the memory is shared among them and we can see it by running the workflow:

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

It is important to note, though, the resources are preserved across steps of the same workflow instance, but not across different workflows. If we were to run two `WorkflowWithMemory` instances, their memories would be separate and independent:

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

## Using Steteless Resources

Resources can also be stateless, meaning that we can configure them *not* to be preserved across steps in the same run.

In order to do so, we just need to specify `cache=False` when instantiating `Resource` - let's see this in a simple example, using a custom `Counter` class:

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

Now that we've mastered resources, let's take a look at [observability and debugging](./observability.md) in workflows.
