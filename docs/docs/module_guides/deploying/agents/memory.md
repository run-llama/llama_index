# Memory

## Concept

Memory is a core component of agentic systems. It allows you to store and retrieve information from the past.

In LlamaIndex, you can typically customize memory by using an existing `BaseMemory` class, or by creating a custom one.

As the agent runs, it will make calls to `memory.put()` to store information, and `memory.get()` to retrieve information.

By default, the `ChatMemoryBuffer` is used across the framework, to create a basic buffer of chat history, which gives the agent the last X messages that fit into a token limit.

## Usage

### Setting Memory Types for an Agent

You can set the memory for an agent by passing it into the `run()` method:

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=40000)

agent = FunctionAgent(llm=llm, tools=tools)

response = await agent.run("<question that invokes tool>", memory=memory)
```

### Managing the Memory Manually

You can also manage the memory manually by calling `memory.put()` and `memory.get()` directly, and passing in the chat history:

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer


memory = ChatMemoryBuffer.from_defaults(token_limit=40000)
memory.put(ChatMessage(role="user", content="Hello, world!"))
memory.put(ChatMessage(role="assistant", content="Hello, world to you too!"))
chat_history = memory.get()

agent = FunctionAgent(llm=llm, tools=tools)

# passing in the chat history overrides any existing memory
response = await agent.run(
    "<question that invokes tool>", chat_history=chat_history
)
```

### Retrieving the Latest Memory from an Agent

You can get the latest memory from an agent by grabbing it from the agent context:

```python
from llama_index.core.workflow import Context

ctx = Context(agent)

response = await ctx.run("<question that invokes tool>", ctx=ctx)

# get the memory
memory = await ctx.get("memory")
chat_history = memory.get_all()
```

## Memory Types

In `llama_index.core.memory`, we offer a few different memory types:

- `ChatMemoryBuffer`: A basic memory buffer that stores the last X messages that fit into a token limit.
- `ChatSummaryMemoryBuffer`: A memory buffer that stores the last X messages that fit into a token limit, and also summarizes the conversation periodically when it gets too long.
- `VectorMemory`: A memory that stores and retrieves chat messages from a vector database. It makes no guarantees about the order of the messages, and returns the most similar messages to the latest user message.
- `SimpleComposableMemory`: A memory that composes multiple memories together. Usually used to combine `VectorMemory` with `ChatMemoryBuffer` or `ChatSummaryMemoryBuffer`.

## Examples

You can find a few examples of memory in action below:

- [Chat Memory Buffer](../../../examples/agent/memory/chat_memory_buffer.ipynb)
- [Chat Summary Memory Buffer](../../../examples/agent/memory/summary_memory_buffer.ipynb)
- [Composable Memory](../../../examples/agent/memory/composable_memory.ipynb)
- [Vector Memory](../../../examples/agent/memory/vector_memory.ipynb)
