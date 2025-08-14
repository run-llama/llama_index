# Customizing entry and exit points

Most of the times, relying on the default entry and exit points we have seen in the [Getting Started] section is enough.
However, workflows support custom events where you normally would expect `StartEvent` and `StopEvent`, let's see how.

## Using a custom `StartEvent`

When we call the `run()` method on a workflow instance, the keyword arguments passed become fields of a `StartEvent`
instance that's automatically created under the hood. In case we want to pass complex data to start a workflow, this
approach might become cumbersome, and it's when we can introduce a custom start event.

To be able to use a custom start event, the first step is creating a custom class that inherits from `StartEvent`:

```python
from pathlib import Path

from llama_index.core.workflow import StartEvent
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.llms.openai import OpenAI


class MyCustomStartEvent(StartEvent):
    a_string_field: str
    a_path_to_somewhere: Path
    an_index: LlamaCloudIndex
    an_llm: OpenAI
```

All we have to do now is using `MyCustomStartEvent` as event type in the steps that act as entry points.
Take this artificially complex step for example:

```python
class JokeFlow(Workflow):
    ...

    @step
    async def generate_joke_from_index(
        self, ev: MyCustomStartEvent
    ) -> JokeEvent:
        # Build a query engine using the index and the llm from the start event
        query_engine = ev.an_index.as_query_engine(llm=ev.an_llm)
        topic = query_engine.query(
            f"What is the closest topic to {a_string_field}"
        )
        # Use the llm attached to the start event to instruct the model
        prompt = f"Write your best joke about {topic}."
        response = await ev.an_llm.acomplete(prompt)
        # Dump the response on disk using the Path object from the event
        ev.a_path_to_somewhere.write_text(str(response))
        # Finally, pass the JokeEvent along
        return JokeEvent(joke=str(response))
```

We could still pass the fields of `MyCustomStartEvent` as keyword arguments to the `run` method of our workflow, but
that would be, again, cumbersome. A better approach is to use pass the event instance through the `start_event`
keyword argument like this:

```python
custom_start_event = MyCustomStartEvent(...)
w = JokeFlow(timeout=60, verbose=False)
result = await w.run(start_event=custom_start_event)
print(str(result))
```

This approach makes the code cleaner and more explicit and allows autocompletion in IDEs to work properly.

## Using a custom `StopEvent`

Similarly to `StartEvent`, relying on the built-in `StopEvent` works most of the times but not always. In fact, when we
use `StopEvent`, the result of a workflow must be set to the `result` field of the event instance. Since a result can
be any Python object, the `result` field of `StopEvent` is typed as `Any`, losing any advantage from the typing system.
Additionally, returning more than one object is cumbersome: we usually stuff a bunch of unrelated objects into a
dictionary that we then assign to `StopEvent.result`.

First step to support custom stop events, we need to create a subclass of `StopEvent`:

```python
from llama_index.core.workflow import StopEvent


class MyStopEvent(StopEvent):
    critique: CompletionResponse
```

We can now replace `StopEvent` with `MyStopEvent` in our workflow:

```python
class JokeFlow(Workflow):
    ...

    @step
    async def critique_joke(self, ev: JokeEvent) -> MyStopEvent:
        joke = ev.joke

        prompt = f"Give a thorough analysis and critique of the following joke: {joke}"
        response = await self.llm.acomplete(prompt)
        return MyStopEvent(response)

    ...
```

The one important thing we need to remember when using a custom stop events, is that the result of a workflow run
will be the instance of the event:

```python
w = JokeFlow(timeout=60, verbose=False)
# Warning! `result` now contains an instance of MyStopEvent!
result = await w.run(topic="pirates")
# We can now access the event fields as any normal Event
print(result.critique.text)
```

This approach takes advantage of the Python typing system, is friendly to autocompletion in IDEs and allows
introspection from outer applications that now know exactly what a workflow run will return.
