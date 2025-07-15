# Using Structured Output

Agents are knowingly unreliable in their outputs, as most of the times we would like them to answer in a specific format but they do not follow the rules due to their internal complexity.

To tackle this challenge, we added the possibility of specifying a structured output format within single and multi-agent workflows so that you can have a more fine-grained control over the data structure that your agent returns.

You can do it by adding an `output_cls` argument to the initialization options of your agent/multi-agent workflows:

```python
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field

llm = OpenAI(model="gpt-4.1")

# single agent


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

# multi agent


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

# multi agent from tools or functions

## define multi-agent workflow
workflow = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=[get_weather],
    system_prompt="You are a weather agent that can get the weather for a given location",
    output_cls=Weather,
    llm=llm,
)
```

Or you can define a custom function that takes as input a sequence of ChatMessages produced by the agent workflow and returns a dictionary (that can be turned into a BaseModel subclass):

```python
import json
from llama_index.core.llms import ChatMessage
from typing import List, Dict, Any


class Flavor(BaseModel):
    flavor: str
    with_sugar: bool


# define it sync
def structured_output_parsing(messages: List[ChatMessage]) -> Dict[str, Any]:
    sllm = llm.as_structured_llm(Flavor)
    messages.append(
        ChatMessage(
            role="user",
            content="Given the previous message history, structure the output based on the provided format.",
        )
    )
    response = sllm.chat(messages)
    return json.loads(response.message.content)


# define it async
async def astructured_output_parsing(
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


agent_sync = FunctionAgent(
    tools=[get_flavor],
    name="ice_cream_shopper",
    system_prompt="You are an agent that knows the ice cream flavors in various shops.",
    structured_output_fn=structured_output_parsing,
    llm=llm,
)
agent_async = FunctionAgent(
    tools=[get_flavor],
    name="ice_cream_shopper",
    system_prompt="You are an agent that knows the ice cream flavors in various shops.",
    structured_output_fn=astructured_output_parsing,
    llm=llm,
)
```

Now you can get the structured output while the workflow by using the `AgentStreamStructuredOutput` event:

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

And you can parse the structured output in the agent's response accessing it directly as a dictionary or loading it as a BaseModel subclass by using the `get_pydantic_model` method:

```python
print(response.structured_response)
print(response.get_pydantic_model(Flavor))
```
