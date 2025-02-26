# Human in the loop

Tools can also be defined that get a human in the loop. This is useful for tasks that require human input, such as confirming a tool call or providing feedback.

As we'll see in the our [Workflows tutorial](../workflows/index.md), the way Workflows work under the hood of AgentWorkflow is by running steps which both emit and receive events. Here's a diagram of the steps (in blue) that makes up an AgentWorkflow and the events (in green) that pass data between them. You'll recognize these events, they're the same ones we were handling in the output stream earlier.

![Workflows diagram](./agentworkflow.jpg)

To get a human in the loop, we'll get our tool to emit an event that isn't received by any other step in the workflow. We'll then tell our tool to wait until it receives a specific "reply" event.

We have built-in `InputRequiredEvent` and `HumanResponseEvent` events to use for this purpose. If you want to capture different forms of human input, you can subclass these events to match your own preferences. Let's import them:

```python
from llama_index.core.workflow import (
    InputRequiredEvent,
    HumanResponseEvent,
)
```

Next we'll create a tool that performs a hypothetical dangerous task. There are a couple of new things happening here:

* We're calling `write_event_to_stream` with an `InputRequiredEvent`. This emits an event to the external stream to be captured. You can attach arbitrary data to the event, which we do in the form of a `user_name`.
* We call `wait_for_event`, specifying that we want to wait for a `HumanResponseEvent` and that it must have the `user_name` set to "Laurie". You can see how this would be useful in a multi-user system where more than one incoming event might be involved.

```python
async def dangerous_task(ctx: Context) -> str:
    """A dangerous task that requires human confirmation."""

    # emit an event to the external stream to be captured
    ctx.write_event_to_stream(
        InputRequiredEvent(
            prefix="Are you sure you want to proceed? ",
            user_name="Laurie",
        )
    )

    # wait until we see a HumanResponseEvent
    response = await ctx.wait_for_event(
        HumanResponseEvent, requirements={"user_name": "Laurie"}
    )

    # act on the input from the event
    if response.response.strip().lower() == "yes":
        return "Dangerous task completed successfully."
    else:
        return "Dangerous task aborted."
```

We create our agent as usual, passing it the tool we just defined:

```python
workflow = AgentWorkflow.from_tools_or_functions(
    [dangerous_task],
    llm=llm,
    system_prompt="You are a helpful assistant that can perform dangerous tasks.",
)
```

Now we can run the workflow, handling the `InputRequiredEvent` just like any other streaming event, and responding with a `HumanResponseEvent` passed in using the `send_event` method:

```python
handler = workflow.run(user_msg="I want to proceed with the dangerous task.")

async for event in handler.stream_events():
    if isinstance(event, InputRequiredEvent):
        # capture keyboard input
        response = input(event.prefix)
        # send our response back
        handler.ctx.send_event(
            HumanResponseEvent(
                response=response,
                user_name=event.user_name,
            )
        )

response = await handler
print(str(response))
```

As usual, you can see the [full code of this example](https://github.com/run-llama/python-agents-tutorial/blob/main/5_human_in_the_loop.py).

You can do anything you want to capture the input; you could use a GUI, or audio input, or even get another, separate agent involved. If your input is going to take a while, or happen in another process, you might want to [serialize the context](./state.md) and save it to a database or file so that you can resume the workflow later.

Speaking of getting other agents involved brings us to our next section, [multi-agent systems](./multi_agent.md).
