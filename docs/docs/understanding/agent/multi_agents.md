# Multi-Agent Workflows

The AgentWorkflow uses Workflow Agents to allow you to create a system of multiple agents that can collaborate and hand off tasks to each other based on their specialized capabilities. This enables building more complex agent systems where different agents handle different aspects of a task.

## Quick Start

Here's a simple example of setting up a multi-agent workflow with a calculator agent and a retriever agent:

```python
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReactAgent,
)
from llama_index.core.tools import FunctionTool


# Define some tools
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


# Create agent configs
# NOTE: we can use FunctionAgent or ReactAgent here.
# FunctionAgent works for LLMs with a function calling API.
# ReactAgent works for any LLM.
calculator_agent = FunctionAgent(
    name="calculator",
    description="Performs basic arithmetic operations",
    system_prompt="You are a calculator assistant.",
    tools=[
        FunctionTool.from_defaults(fn=add),
        FunctionTool.from_defaults(fn=subtract),
    ],
    llm=OpenAI(model="gpt-4"),
)

retriever_agent = FunctionAgent(
    name="retriever",
    description="Manages data retrieval",
    system_prompt="You are a retrieval assistant.",
    llm=OpenAI(model="gpt-4"),
)

# Create and run the workflow
workflow = AgentWorkflow(
    agents=[calculator_agent, retriever_agent], root_agent="calculator"
)

# Run the system
response = await workflow.run(user_msg="Can you add 5 and 3?")

#  Or stream the events
handler = workflow.run(user_msg="Can you add 5 and 3?")
async for event in handler.stream_events():
    if hasattr(event, "delta"):
        print(event.delta, end="", flush=True)
```

## How It Works

The AgentWorkflow manages a collection of agents, each with their own specialized capabilities. One agent must be designated as the root agent in the `AgentWorkflow` constructor.

When a user message comes in, it's first routed to the root agent. Each agent can then:

1. Handle the request directly using their tools
2. Hand off to another agent better suited for the task
3. Return a response to the user

## Configuration Options

### Agent Config

Each agent holds a certain set of configuration options. Whether you use `FunctionAgent` or `ReactAgent`, the core options are the same.

```python
FunctionAgent(
    # Unique name for the agent (str)
    name="name",
    # Description of agent's capabilities (str)
    description="description",
    # System prompt for the agent (str)
    system_prompt="system_prompt",
    # Tools available to this agent (List[BaseTool])
    tools=[...],
    # LLM to use for this agent. (BaseLLM)
    llm=OpenAI(model="gpt-4"),
    # List of agents this one can hand off to. Defaults to all agents. (List[str])
    can_handoff_to=[...],
)
```

### Workflow Options

The AgentWorkflow constructor accepts:

```python
AgentWorkflow(
    # List of agent configs. (List[BaseWorkflowAgent])
    agents=[...],
    # Root agent name. (str)
    root_agent="root_agent",
    # Initial state dict. (Optional[dict])
    initial_state=None,
    # Custom prompt for handoffs. Should contain the `agent_info` string variable. (Optional[str])
    handoff_prompt=None,
    # Custom prompt for state. Should contain the `state` and `msg` string variables. (Optional[str])
    state_prompt=None,
)
```

### State Management

#### Initial Global State

You can provide an initial state dict that will be available to all agents:

```python
workflow = AgentWorkflow(
    agents=[...],
    root_agent="root_agent",
    initial_state={"counter": 0},
    state_prompt="Current state: {state}. User message: {msg}",
)
```

The state is stored in the `state` key of the workflow context.

#### Persisting State Between Runs

In order to persist state between runs, you can pass in the context from the previous run:

```python
workflow = AgentWorkflow(...)

# Run the workflow
handler = workflow.run(user_msg="Can you add 5 and 3?")
response = await handler

# Pass in the context from the previous run
handler = workflow.run(ctx=handler.ctx, user_msg="Can you add 5 and 3?")
response = await handler
```

#### Serializing Context / State

As with normal workflows, the context is serializable:

```python
from llama_index.core.workflow import (
    Context,
    JsonSerializer,
    JsonPickleSerializer,
)

# the default serializer is JsonSerializer for safety
ctx_dict = handler.ctx.to_dict(serializer=JsonSerializer())

# then you can rehydrate the context
ctx = Context.from_dict(ctx_dict, serializer=JsonSerializer())
```

## Streaming Events

The workflow emits various events during execution that you can stream:

```python
async for event in workflow.run(...).stream_events():
    if isinstance(event, AgentInput):
        print(event.input)
        print(event.current_agent_name)
    elif isinstance(event, AgentStream):
        # Agent thinking/tool calling response stream
        print(event.delta)
        print(event.current_agent_name)
    elif isinstance(event, AgentOutput):
        print(event.response)
        print(event.tool_calls)
        print(event.raw)
        print(event.current_agent_name)
    elif isinstance(event, ToolCall):
        # Tool being called
        print(event.tool_name)
        print(event.tool_kwargs)
    elif isinstance(event, ToolCallResult):
        # Result of tool call
        print(event.tool_output)
```

## Accessing Context in Tools

The `FunctionToolWithContext` allows tools to access the workflow context:

```python
from llama_index.core.workflow import FunctionToolWithContext


async def get_counter(ctx: Context) -> int:
    """Get the current counter value."""
    return await ctx.get("counter", default=0)


counter_tool = FunctionToolWithContext.from_defaults(
    async_fn=get_counter, description="Get the current counter value"
)
```

## Human in the Loop

Using the context, you can implement a human in the loop pattern in your tools:

```python
from llama_index.core.workflow import Event


class AskForConfirmationEvent(Event):
    """Ask for confirmation event."""

    confirmation_id: str


class ConfirmationEvent(Event):
    """Confirmation event."""

    confirmation: bool
    confirmation_id: str


async def ask_for_confirmation(ctx: Context) -> bool:
    """Ask the user for confirmation."""
    ctx.write_event_to_stream(AskForConfirmationEvent(confirmation_id="1234"))

    result = await ctx.wait_for_event(
        ConfirmationEvent, requirements={"confirmation_id": "1234"}
    )
    return result.confirmation
```

When this function is called, it will block the workflow execution until the user sends the required confirmation event.

```python
handler = workflow.run(user_msg="Can you add 5 and 3?")

async for event in handler.stream_events():
    if isinstance(event, AskForConfirmationEvent):
        print(event.confirmation_id)
        handler.ctx.send_event(
            ConfirmationEvent(confirmation=True, confirmation_id="1234")
        )
    ...
```
