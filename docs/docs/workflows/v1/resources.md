# Resource Objects

Resources are external dependencies you can inject into the steps of a workflow.

As a simple example, look at `memory` in the following workflow:

```python
from llama_index.core.workflow.resource import Resource
from llama_index.core.memory import Memory


def get_memory(*args, **kwargs):
    return Memory.from_defaults("user_id_123", token_limit=60000)


class SecondEvent(Event):
    msg: str


class WorkflowWithResource(Workflow):
    @step
    async def first_step(
        self,
        ev: StartEvent,
        memory: Annotated[Memory, Resource(get_memory)],
    ) -> SecondEvent:
        print("Memory before step 1", memory)
        await memory.aput(
            ChatMessage(role="user", content="This is the first step")
        )
        print("Memory after step 1", memory)
        return SecondEvent(msg="This is an input for step 2")

    @step
    async def second_step(
        self, ev: SecondEvent, memory: Annotated[Memory, Resource(get_memory)]
    ) -> StopEvent:
        print("Memory before step 2", memory)
        await memory.aput(ChatMessage(role="user", content=ev.msg))
        print("Memory after step 2", memory)
        return StopEvent(result="Messages put into memory")
```

To inject a resource into a workflow step, you have to add a parameter to the step signature and define its type,
using `Annotated` and invoke the `Resource()` wrapper passing a function or callable returning the actual Resource
object. The return type of the wrapped function must match the declared type, ensuring consistency between what’s
expected and what’s provided during execution. In the example above, `memory: Annotated[Memory, Resource(get_memory)`
defines a resource of type `Memory` that will be provided by the `get_memory()` function and passed to the step in the
`memory` parameter when the workflow runs.

Resources are shared among steps of a workflow, and the `Resource()` wrapper will invoke the factory function only once.
In case this is not the desired behavior, passing `cache=False` to `Resource()` will inject different resource objects
in different steps, invoking the factory function as many times.
