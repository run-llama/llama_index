# Streaming events

Workflows can be complex -- they are designed to handle complex, branching, concurrent logic -- which means they can take time to fully execute. To provide your user with a good experience, you may want to provide an indication of progress by streaming events as they occur. Workflows have built-in support for this on the `Context` object.

To get this done, let's bring in all the deps we need:

```python
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context,
)
import asyncio
from llama_index.llms.openai import OpenAI
from llama_index.utils.workflow import draw_all_possible_flows
```

Let's set up some events for a simple three-step workflow, plus an event to handle streaming our progress as we go:

```python
class FirstEvent(Event):
    first_output: str


class SecondEvent(Event):
    second_output: str
    response: str


class ProgressEvent(Event):
    msg: str
```

And define a workflow class that sends events:

```python
class MyWorkflow(Workflow):
    @step
    async def step_one(self, ctx: Context, ev: StartEvent) -> FirstEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="Step one is happening"))
        return FirstEvent(first_output="First step complete.")

    @step
    async def step_two(self, ctx: Context, ev: FirstEvent) -> SecondEvent:
        llm = OpenAI(model="gpt-4o-mini")
        generator = await llm.astream_complete(
            "Please give me the first 3 paragraphs of Moby Dick, a book in the public domain."
        )
        async for response in generator:
            # Allow the workflow to stream this piece of response
            ctx.write_event_to_stream(ProgressEvent(msg=response.delta))
        return SecondEvent(
            second_output="Second step complete, full response attached",
            response=str(response),
        )

    @step
    async def step_three(self, ctx: Context, ev: SecondEvent) -> StopEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="Step three is happening"))
        return StopEvent(result="Workflow complete.")
```

!!! tip
    `OpenAI()` here assumes you have an `OPENAI_API_KEY` set in your environment. You could also pass one in using the `api_key` parameter.

In `step_one` and `step_three` we write individual events to the event stream. In `step_two` we use `astream_complete` to produce an iterable generator of the LLM's response, then we produce an event for each chunk of data the LLM sends back to us -- roughly one per word -- before returning the final response to `step_three`.

To actually get this output, we need to run the workflow asynchronously and listen for the events, like this:

```python
async def main():
    w = MyWorkflow(timeout=30, verbose=True)
    handler = w.run(first_input="Start the workflow.")

    async for ev in handler.stream_events():
        if isinstance(ev, ProgressEvent):
            print(ev.msg)

    final_result = await handler
    print("Final result", final_result)

    draw_all_possible_flows(MyWorkflow, filename="streaming_workflow.html")


if __name__ == "__main__":
    asyncio.run(main())
```

`run` runs the workflow in the background, while `stream_events` will provide any event that gets written to the stream. It stops when the stream delivers a `StopEvent`, after which you can get the final result of the workflow as you normally would.


Next let's look at [concurrent execution](concurrent_execution.md).
