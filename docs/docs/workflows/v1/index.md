# Getting Started

!!! tip
    Workflows make async a first-class citizen, and this page assumes you are running in an async environment. What this means for you is setting up your code for async properly. If you are already running in a server like FastAPI, or in a notebook, you can freely use await already!

    If you are running your own python scripts, its best practice to have a single async entry point.

    ```python
    async def main():
        w = MyWorkflow(...)
        result = await w.run(...)
        print(result)


    if __name__ == "__main__":
        import asyncio

        asyncio.run(main())
    ```


As an illustrative example, let's consider a naive workflow where a joke is generated and then critiqued.

```python
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

# `pip install llama-index-llms-openai` if you don't already have it
from llama_index.llms.openai import OpenAI


class JokeEvent(Event):
    joke: str


class JokeFlow(Workflow):
    llm = OpenAI()

    @step
    async def generate_joke(self, ev: StartEvent) -> JokeEvent:
        topic = ev.topic

        prompt = f"Write your best joke about {topic}."
        response = await self.llm.acomplete(prompt)
        return JokeEvent(joke=str(response))

    @step
    async def critique_joke(self, ev: JokeEvent) -> StopEvent:
        joke = ev.joke

        prompt = f"Give a thorough analysis and critique of the following joke: {joke}"
        response = await self.llm.acomplete(prompt)
        return StopEvent(result=str(response))


w = JokeFlow(timeout=60, verbose=False)
result = await w.run(topic="pirates")
print(str(result))
```

There's a few moving pieces here, so let's go through this piece by piece.

## Defining Workflow Events

```python
class JokeEvent(Event):
    joke: str
```

Events are user-defined pydantic objects. You control the attributes and any other auxiliary methods. In this case, our workflow relies on a single user-defined event, the `JokeEvent`.

## Setting up the Workflow Class

```python
class JokeFlow(Workflow):
    llm = OpenAI(model="gpt-4o-mini")
    ...
```

Our workflow is implemented by subclassing the `Workflow` class. For simplicity, we attached a static `OpenAI` llm instance.

## Workflow Entry Points

```python
class JokeFlow(Workflow):
    ...

    @step
    async def generate_joke(self, ev: StartEvent) -> JokeEvent:
        topic = ev.topic

        prompt = f"Write your best joke about {topic}."
        response = await self.llm.acomplete(prompt)
        return JokeEvent(joke=str(response))

    ...
```

Here, we come to the entry-point of our workflow. While most events are use-defined, there are two special-case events,
the `StartEvent` and the `StopEvent` that the framework provides out of the box. Here, the `StartEvent` signifies where
to send the initial workflow input.

The `StartEvent` is a bit of a special object since it can hold arbitrary attributes. Here, we accessed the topic with
`ev.topic`, which would raise an error if it wasn't there. You could also do `ev.get("topic")` to handle the case where
the attribute might not be there without raising an error.

At this point, you may have noticed that we haven't explicitly told the workflow what events are handled by which steps.
Instead, the `@step` decorator is used to infer the input and output types of each step. Furthermore, these inferred
input and output types are also used to verify for you that the workflow is valid before running!

## Workflow Exit Points

```python
class JokeFlow(Workflow):
    ...

    @step
    async def critique_joke(self, ev: JokeEvent) -> StopEvent:
        joke = ev.joke

        prompt = f"Give a thorough analysis and critique of the following joke: {joke}"
        response = await self.llm.acomplete(prompt)
        return StopEvent(result=str(response))

    ...
```

Here, we have our second, and last step, in the workflow. We know its the last step because the special `StopEvent` is
returned. When the workflow encounters a returned `StopEvent`, it immediately stops the workflow and returns whatever
we passed in the `result` parameter.

In this case, the result is a string, but it could be a dictionary, list, or any other object.

## Running the Workflow

```python
w = JokeFlow(timeout=60, verbose=False)
result = await w.run(topic="pirates")
print(str(result))
```

Lastly, we create and run the workflow. There are some settings like timeouts (in seconds) and verbosity to help with
debugging.

The `.run()` method is async, so we use await here to wait for the result. The keyword arguments passed to `run()` will
become fields of the special `StartEvent` that will be automatically emitted and start the workflow. As we have seen,
in this case `topic` will be accessed from the step with `ev.topic`.

## Decorating non-class Functions

You can also decorate and attach steps to a workflow without subclassing it.

Below is the `JokeFlow` from earlier, but defined without subclassing.

```python
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.llms.openai import OpenAI


class JokeEvent(Event):
    joke: str


joke_flow = Workflow(timeout=60, verbose=True)


@step(workflow=joke_flow)
async def generate_joke(ev: StartEvent) -> JokeEvent:
    topic = ev.topic

    prompt = f"Write your best joke about {topic}."

    llm = OpenAI()
    response = await llm.acomplete(prompt)
    return JokeEvent(joke=str(response))


@step(workflow=joke_flow)
async def critique_joke(ev: JokeEvent) -> StopEvent:
    joke = ev.joke

    prompt = (
        f"Give a thorough analysis and critique of the following joke: {joke}"
    )
    response = await llm.acomplete(prompt)
    return StopEvent(result=str(response))
```

## Learn More

- [Customizing Entry/Exit Points](./customizing_entry_exit_points.md)
- [Drawing Workflows](./drawing.md)
- [Managing Events](./managing_events.md)
- [Managing State](./managing_state.md)
- [Injecting External Resources](./resources.md)
- [Handling Errors](./retry_steps.md)
- [Stepwise Execution](./stepwise.md)
- [Deploying with LlamaDeploy](./deployment.md)

## Examples

To help you become more familiar with the workflow concept and its features, LlamaIndex documentation offers example
notebooks that you can run for hands-on learning:

- [Common Workflow Patterns](../../examples/workflow/workflows_cookbook.ipynb) walks you through common usage patterns
like looping and state management using simple workflows. It's usually a great place to start.
- [RAG + Reranking](../../examples/workflow/rag.ipynb) shows how to implement a real-world use case with a fairly
simple workflow that performs both ingestion and querying.
- [Citation Query Engine](../../examples/workflow/citation_query_engine.ipynb) similar to RAG + Reranking, the
notebook focuses on how to implement intermediate steps in between retrieval and generation. A good example of how to
use the [`Context`](./managing_state.md#maintaining-context-across-runs) object in a workflow.
- [Corrective RAG](../../examples/workflow/corrective_rag_pack.ipynb) adds some more complexity on top of a RAG
workflow, showcasing how to query a web search engine after an evaluation step.
- [Utilizing Concurrency](../../examples/workflow/parallel_execution.ipynb) explains how to manage the parallel
execution of steps in a workflow, something that's important to know as your workflows grow in complexity.

RAG applications are easy to understand and offer a great opportunity to learn the basics of workflows. However, more complex agentic scenarios involving tool calling, memory, and routing are where workflows excel.

The examples below highlight some of these use-cases.

- [ReAct Agent](../../examples/workflow/react_agent.ipynb) is obviously the perfect example to show how to implement
tools in a workflow.
- [Function Calling Agent](../../examples/workflow/function_calling_agent.ipynb) is a great example of how to use the
LlamaIndex framework primitives in a workflow, keeping it small and tidy even in complex scenarios like function
calling.
- [CodeAct Agent](../../examples/agent/from_scratch_code_act_agent.ipynb) is a great example of how to create a CodeAct Agent from scratch.
- [Human In The Loop: Story Crafting](../../examples/workflow/human_in_the_loop_story_crafting.ipynb) is a powerful
example showing how workflow runs can be interactive and stateful. In this case, to collect input from a human.
- [Reliable Structured Generation](../../examples/workflow/reflection.ipynb) shows how to implement loops in a
workflow, in this case to improve structured output through reflection.
- [Query Planning with Workflows](../../examples/workflow/planning_workflow.ipynb) is an example of a workflow
that plans a query by breaking it down into smaller items, and executing those smaller items. It highlights how
to stream events from a workflow, execute steps in parallel, and looping until a condition is met.
- [Checkpointing Workflows](../../examples/workflow/checkpointing_workflows.ipynb) is a more exhaustive demonstration of how to make full use of `WorkflowCheckpointer` to checkpoint Workflow runs.

Last but not least, a few more advanced use cases that demonstrate how workflows can be extremely handy if you need
to quickly implement prototypes, for example from literature:

- [Advanced Text-to-SQL](../../examples/workflow/advanced_text_to_sql.ipynb)
- [JSON Query Engine](../../examples/workflow/JSONalyze_query_engine.ipynb)
- [Long RAG](../../examples/workflow/long_rag_pack.ipynb)
- [Multi-Step Query Engine](../../examples/workflow/multi_step_query_engine.ipynb)
- [Multi-Strategy Workflow](../../examples/workflow/multi_strategy_workflow.ipynb)
- [Router Query Engine](../../examples/workflow/router_query_engine.ipynb)
- [Self Discover Workflow](../../examples/workflow/self_discover_workflow.ipynb)
- [Sub-Question Query Engine](../../examples/workflow/sub_question_query_engine.ipynb)
