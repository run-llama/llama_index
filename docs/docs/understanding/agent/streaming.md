# Streaming output and events

In real-world use, agents can take a long time to run. Providing feedback to the user about the progress of the agent is critical, and streaming allows you to do that.

`AgentWorkflow` provides a set of pre-built events that you can use to stream output to the user. Let's take a look at how that's done.

First, we're going to introduce a new tool that takes some time to execute. In this case we'll use a web search tool called [Tavily](https://llamahub.ai/l/tools/llama-index-tools-tavily-research), which is available in LlamaHub.

```bash
pip install llama-index-tools-tavily-research
```

It requires an API key, which we're going to set in our `.env` file as `TAVILY_API_KEY` and retrieve using the `os.getenv` method. Let's bring in our imports:

```python
from llama_index.tools.tavily_research import TavilyToolSpec
import os
```

And initialize the tool:

```python
tavily_tool = TavilyToolSpec(api_key=os.getenv("TAVILY_API_KEY"))
```

Now we'll create an agent using that tool and an LLM that we initialized just like we did previously.

```python
workflow = AgentWorkflow.from_tools_or_functions(
    tavily_tool.to_tool_list(),
    llm=llm,
    system_prompt="You're a helpful assistant that can search the web for information.",
)
```

In previous examples, we've used `await` on the `workflow.run` method to get the final response from the agent. However, if we don't await the response, we get an asynchronous iterator back that we can iterate over to get the events as they come in. This iterator will return all sorts of events. We'll start with an `AgentStream` event, which contains the "delta" (the most recent change) to the output as it comes in. We'll need to import that event type:

```python
from llama_index.core.agent.workflow import AgentStream
```

And now we can run the workflow and look for events of that type to output:

```python
handler = workflow.run(user_msg="What's the weather like in San Francisco?")

async for event in handler.stream_events():
    if isinstance(event, AgentStream):
        print(event.delta, end="", flush=True)
```

If you run this yourself, you will see the output arriving in chunks as the agent runs, returning something like this:

```
The current weather in San Francisco is as follows:

- **Temperature**: 17.2°C (63°F)
- **Condition**: Sunny
- **Wind**: 6.3 mph (10.1 kph) from the NNW
- **Humidity**: 54%
- **Pressure**: 1021 mb (30.16 in)
- **Visibility**: 16 km (9 miles)

For more details, you can check the full report [here](https://www.weatherapi.com/).
```

`AgentStream` is just one of many events that `AgentWorkflow` emits as it runs. The others are:

* `AgentInput`: the full message object that begins the agent's execution
* `AgentOutput`: the response from the agent
* `ToolCall`: which tools were called and with what arguments
* `ToolCallResult`: the result of a tool call

You can see us filtering for more of these events in the [full code of this example](https://github.com/run-llama/python-agents-tutorial/blob/main/4_streaming.py).

Next you'll learn about how to get a [human in the loop](./human_in_the_loop.md) to provide feedback to your agents.
