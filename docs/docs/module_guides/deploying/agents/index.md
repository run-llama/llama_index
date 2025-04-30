# Agents

In LlamaIndex, we define an "agent" as a specific system that uses an LLM, memory, and tools, to handle inputs from outside users. Contrast this with the term "agentic", which generally refers to a superclass of agents, which is any system with LLM decision making in the process.

To create an agent in LlamaIndex, it takes only a few lines of code:

```python
import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI


# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


# Create an agent workflow with our calculator tool
agent = FunctionAgent(
    tools=[multiply],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant that can multiply two numbers.",
)


async def main():
    # Run the agent
    response = await agent.run("What is 1234 * 4567?")
    print(str(response))


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
```

Calling this agent kicks off a specific loop of actions:

- Agent gets the latest message + chat history
- The tool schemas and chat history get sent over the API
- The Agent responds either with a direct response, or a list of tool calls
    - Every tool call is executed
    - The tool call results are added to the chat history
    - The Agent is invoked again with updated history, and either responds directly or selects more calls

## Tools

Tools can be defined simply as python functions, or further customized using classes like `FunctionTool` and `QueryEngineTool`. LlamaIndex also provides sets of pre-defined tools for common APIs using something called `Tool Specs`.

You can read more about configuring tools in the [tools guide](./tools.md)

## Memory

Memory is a core-component when building agents. By default, all LlamaIndex agents are using a ChatMemoryBuffer for memory.

To customize it, you can declare it outside the agent and pass it in:

```python
from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=40000)

response = await agent.run(..., memory=memory)
```

You can read more about configuring memory in the [memory guide](./memory.md)

## Multi-Modal Agents

Some LLMs will support multiple modalities, such as images and text. Using chat messages with content blocks, we can pass in images to an agent for reasoning.

For example, imagine you had a screenshot of the [slide from this presentation](https://docs.google.com/presentation/d/1wy3nuO9ezGS4R99mzP3Q3yvrjAkZ26OGI2NjfqtwAaE/edit?usp=sharing).

You can pass this image to an agent for reasoning, and see that it reads the image and acts accordingly.

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o-mini", api_key="sk-...")


def add(a: int, b: int) -> int:
    """Useful for adding two numbers together."""
    return a + b


workflow = FunctionAgent(
    tools=[add],
    llm=llm,
)

msg = ChatMessage(
    role="user",
    blocks=[
        TextBlock(text="Follow what the image says."),
        ImageBlock(path="./screenshot.png"),
    ],
)

response = await workflow.run(msg)
print(str(response))
```

## Multi-Agent Systems

You can combine agents into a multi-agent system, where each agent is able to hand off control to another agent to coordinate while completing tasks.

```python
from llama_index.core.agent.workflow import AgentWorkflow

multi_agent = AgentWorkflow(agents=[FunctionAgent(...), FunctionAgent(...)])

resp = await agent.run("query")
```

Read on to learn more about [multi-agent systems](../../../understanding/agent/multi_agent.md).

## Manual Agents

While the agent classes like `FunctionAgent`, `ReActAgent`, `CodeActAgent`, and `AgentWorkflow` abstract away a lot of details, sometimes its desirable to build your own lower-level agents.

Using the `LLM` objects directly, you can quickly implement a basic agent loop, while having full control over how the tool calling and error handling works.

```python
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI


def select_song(song_name: str) -> str:
    """Useful for selecting a song."""
    return f"Song selected: {song_name}"


tools = [FunctionTool.from_defaults(select_song)]
tools_by_name = {t.metadata.name: t for t in [tool]}

# call llm with initial tools + chat history
chat_history = [ChatMessage(role="user", content="Pick a random song for me")]
resp = llm.chat_with_tools([tool], chat_history=chat_history)

# parse tool calls from response
tool_calls = llm.get_tool_calls_from_response(
    resp, error_on_no_tool_call=False
)

# loop while there are still more tools to call
while tool_calls:
    # add the LLM's response to the chat history
    chat_history.append(resp.message)

    # call every tool and add its result to chat_history
    for tool_call in tool_calls:
        tool_name = tool_call.tool_name
        tool_kwargs = tool_call.tool_kwargs

        print(f"Calling {tool_name} with {tool_kwargs}")
        tool_output = tool(**tool_kwargs)
        chat_history.append(
            ChatMessage(
                role="tool",
                content=str(tool_output),
                # most LLMs like OpenAI need to know the tool call id
                additional_kwargs={"tool_call_id": tool_call.tool_id},
            )
        )

        # check if the LLM can write a final response or calls more tools
        resp = llm.chat_with_tools([tool], chat_history=chat_history)
        tool_calls = llm.get_tool_calls_from_response(
            resp, error_on_no_tool_call=False
        )

# print the final response
print(resp.message.content)
```

## Examples / Module Guides

You can find a more complete list of examples and module guides in the [module guides page](./modules.md).
