# Using Structured Output

Most of the time you need results from an agent in a specific format. Agents results can return structured json in two ways:

1. `output_cls` – a Pydantic model to use as a schema for the output
2. `structured_output_fn` – For more advanced use cases, supply a custom function that validates or rewrites the agent’s conversation into any model you want.

Both single-agents like `FunctionAgent` and `ReActAgent`, as well as multi-agent `AgentWorkflow` workflows, support these options - let's explore the possibilities:

### Use `output_cls`

```python
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field

llm = OpenAI(model="gpt-4.1")


## define structured output format  and tools
class MathResult(BaseModel):
    operation: str = Field(description="the performed operation")
    result: int = Field(description="the result of the operation")


def multiply(x: int, y: int):
    """Multiply two numbers"""
    return x * y


## define agent
agent = FunctionAgent(
    tools=[multiply],
    name="calculator",
    system_prompt="You are a calculator agent who can multiply two numbers using the `multiply` tool.",
    output_cls=MathResult,
    llm=llm,
)

response = await agent.run("What is 3415 * 43144?")
print(response.structured_response)
print(response.get_pydantic_model(MathResult))
```

This also works with mutl-agent workflows:

```python
## define structured output format  and tools
class Weather(BaseModel):
    location: str = Field(description="The location")
    weather: str = Field(description="The weather")


def get_weather(location: str):
    """Get the weather for a given location"""
    return f"The weather in {location} is sunny"


## define single agents
agent = FunctionAgent(
    llm=llm,
    tools=[get_weather],
    system_prompt="You are a weather agent that can get the weather for a given location",
    name="WeatherAgent",
    description="The weather forecaster agent.",
)
main_agent = FunctionAgent(
    name="MainAgent",
    tools=[],
    description="The main agent",
    system_prompt="You are the main agent, your task is to dispatch tasks to secondary agents, specifically to WeatherAgent",
    can_handoff_to=["WeatherAgent"],
    llm=llm,
)

## define multi-agent workflow
workflow = AgentWorkflow(
    agents=[main_agent, agent],
    root_agent=main_agent.name,
    output_cls=Weather,
)

response = await workflow.run("What is the weather in Tokyo?")
print(response.structured_response)
print(response.get_pydantic_model(Weather))
```

### Use `structured_output_fn`

The custom function should take as input a sequence of `ChatMessage` objects produced by the agent workflow and returns a dictionary (that can be turned into a `BaseModel` subclass):

```python
import json
from llama_index.core.llms import ChatMessage
from typing import List, Dict, Any


class Flavor(BaseModel):
    flavor: str
    with_sugar: bool


async def structured_output_parsing(
    messages: List[ChatMessage],
) -> Dict[str, Any]:
    sllm = llm.as_structured_llm(Flavor)
    messages.append(
        ChatMessage(
            role="user",
            content="Given the previous message history, structure the output based on the provided format.",
        )
    )
    response = await sllm.achat(messages)
    return json.loads(response.message.content)


def get_flavor(ice_cream_shop: str):
    return "Strawberry with no extra sugar"


agent = FunctionAgent(
    tools=[get_flavor],
    name="ice_cream_shopper",
    system_prompt="You are an agent that knows the ice cream flavors in various shops.",
    structured_output_fn=structured_output_parsing,
    llm=llm,
)

response = await agent.run(
    "What strawberry flavor is available at Gelato Italia?"
)
print(response.structured_response)
print(response.get_pydantic_model(Flavor))
```

### Streaming the Structured Output

You can get the structured output while the workflow is running by using the `AgentStreamStructuredOutput` event:

```python
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStreamStructuredOutput,
)

handler = agent.run("What strawberry flavor is available at Gelato Italia?")

async for event in handler.stream_events():
    if isinstance(event, AgentInput):
        print(event)
    elif isinstance(event, AgentStreamStructuredOutput):
        print(event.output)
        print(event.get_pydantic_model(Weather))
    elif isinstance(event, ToolCallResult):
        print(event)
    elif isinstance(event, ToolCall):
        print(event)
    elif isinstance(event, AgentOutput):
        print(event)
    else:
        pass

response = await handler
```

And you can parse the structured output in the agent's response accessing it directly as a dictionary or loading it as a `BaseModel` subclass by using the `get_pydantic_model` method:

```python
print(response.structured_response)
print(response.get_pydantic_model(Flavor))
```
