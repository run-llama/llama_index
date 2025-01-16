# Multi-Agent Workflows

The `AgentWorkflow` uses Workflow Agents to allow you to create a system of one or more agents that can collaborate and hand off tasks to each other based on their specialized capabilities. This enables building complex agent systems where different agents handle different aspects of a task.

!!! tip
    The `AgentWorkflow` class is built on top of LlamaIndex `Workflows`. For more information on how workflows work, check out the [detailed guide](../../module_guides/workflow/index.md) or [introductory tutorial](../workflows/index.md).

## Quick Start

Here's a simple example of setting up a multi-agent workflow with a calculator agent and a retriever agent:

```python
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
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
# NOTE: we can use FunctionAgent or ReActAgent here.
# FunctionAgent works for LLMs with a function calling API.
# ReActAgent works for any LLM.
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

### Agent Workflow Config

Each agent holds a certain set of configuration options. Whether you use `FunctionAgent` or `ReActAgent`, the core options are the same.

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
    # Timeout for the workflow, in seconds. (Optional[float])
    timeout=None,
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

The state is stored in the `state` key of the workflow context. It will be injected into the `state_prompt` which augments each new user message.

The state can also be modified by tools by accessing the workflow context directly in the tool body.

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

The `FunctionTool` allows tools to access the workflow context if the function has a `Context` type hint as the first parameter:

```python
from llama_index.core.tools import FunctionTool


async def get_counter(ctx: Context) -> int:
    """Get the current counter value."""
    return await ctx.get("counter", default=0)


counter_tool = FunctionToolWithContext.from_defaults(
    async_fn=get_counter, description="Get the current counter value"
)
```

!!! tip
    The `FunctionTool` requires the `ctx` parameter to be passed in explicitly when calling the tool. `AgentWorkflow` will automatically pass in the context for you.

## Human in the Loop

Using the context, you can implement a human in the loop pattern in your tools:

```python
from llama_index.core.workflow import InputRequiredEvent, HumanResponseEvent


async def ask_for_confirmation(ctx: Context) -> bool:
    """Ask the user for confirmation."""
    ctx.write_event_to_stream(
        InputRequiredEvent(prefix="Please confirm", confirmation_id="1234")
    )

    result = await ctx.wait_for_event(
        HumanResponseEvent, requirements={"confirmation_id": "1234"}
    )
    return result.confirmation
```

When this function is called (i.e, when an agent calls this tool), it will block the workflow execution until the user sends the required confirmation event.

```python
handler = workflow.run(user_msg="Can you add 5 and 3?")

async for event in handler.stream_events():
    if isinstance(event, InputRequiredEvent):
        print(event.confirmation_id)
        handler.ctx.send_event(
            HumanResponseEvent(response="True", confirmation_id="1234")
        )
    ...
```

## A Detailed Look at the Workflow

Now that we've covered the basics, let's take a look at how the workflow operates in more detail using an end-to-end example. In this example, assume we have an `AgentWorkflow` with two agents: `generate` and `review`. In this workflow, `generate` is the root agent, and responsible for generating content. The `review` agent is responsible for reviewing the generated content.

When the user sends in a request, here's the actual sequence of events:

1. The workflow initializes the context with:
   - A memory buffer for chat history.
   - The available agents
   - The [initial state](#initial-global-state) dictionary
   - The current agent (initially set to the root agent, `generate`)

2. The user's message is processed:
   - If [state exists](#initial-global-state), it's added to the user's message using the [state prompt](#agent-workflow-config)
   - The message is added to memory
   - The chat history is prepared for the current agent

3. The current agent is set up:
   - The agent's tools are gathered (including any retrieved tools)
   - A special `handoff` tool is added if the agent can hand off to others
   - The agent's system prompt is prepended to the chat history
   - An `AgentInput` event is emitted just before the LLM is called

4. The agent processes the input:
   - The agent generates a response and/or makes tool calls. This generates both `AgentStream` events and an `AgentOutput` event
   - If there are no tool calls, the agent finalizes its response and returns it
   - If there are tool calls, each tool is executed and the results are processed. This will generate a `ToolCall` event and a `ToolCallResult` event for each tool call

5. After tool execution:
   - If any tool was marked as `return_direct=True`, its result becomes the final output
   - If a handoff occurred (via the handoff tool), the workflow switches to the new agent. This will not be added to the chat history in order to maintain the conversation flow.
   - Otherwise, the updated chat history is sent back to the current agent for another step

This cycle continues until either:
- The current agent provides a response without tool calls
- A tool marked as `return_direct=True` is called (except for handoffs)
- The workflow times out (if a timeout was configured)

## Examples

We have a few notebook examples using the `AgentWorkflow` class:

- [Agent Workflow Overview](../../examples/agent/agent_workflow_basic.ipynb)
- [Multi-Agent Research Report Workflow](../../examples/agent/agent_workflow_multi.ipynb)
