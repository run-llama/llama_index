# Memory

## Concept

Memory is a core component of agentic systems. It allows you to store and retrieve information from the past.

In LlamaIndex, you can typically customize memory by using an existing `BaseMemory` class, or by creating a custom one.

As the agent runs, it will make calls to `memory.put()` to store information, and `memory.get()` to retrieve information.

**NOTE:** The `ChatMemoryBuffer` is deprecated. In a future release, the default will be replaced with the `Memory` class, which is more flexible and allows for more complex memory configurations. Examples in this section will use the `Memory` class. By default, the `ChatMemoryBuffer` is used across the framework, to create a basic buffer of chat history, which gives the agent the last X messages that fit into a token limit. The `Memory` class operates similarly, but is more flexible and allows for more complex memory configurations.

## Usage

Using the `Memory` class, you can create a memory that has both short-term memory (i.e. a FIFO queue of messages) and optionally long-term memory (i.e. extracting information over time).

### Configuring Memory for an Agent

You can set the memory for an agent by passing it into the `run()` method:

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import Memory

memory = Memory.from_defaults(session_id="my_session", token_limit=40000)

agent = FunctionAgent(llm=llm, tools=tools)

response = await agent.run("<question that invokes tool>", memory=memory)
```

### Managing the Memory Manually

You can also manage the memory manually by calling `memory.put_messages()` and `memory.get()` directly, and passing in the chat history.

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import Memory


memory = Memory.from_defaults(session_id="my_session", token_limit=40000)
memory.put_messages(
    [
        ChatMessage(role="user", content="Hello, world!"),
        ChatMessage(role="assistant", content="Hello, world to you too!"),
    ]
)
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
chat_history = memory.get()
```

## Customizing Memory

### Short-Term Memory

By default, the `Memory` class will store the last X messages that fit into a token limit. You can customize this by passing in `token_limit` and `chat_history_token_ratio` arguments to the `Memory` class.

- `token_limit` (default: 30000): The maximum number of short-term and long-term tokens to store.
- `chat_history_token_ratio` (default: 0.7): The ratio of tokens in the short-term chat history to the total token limit. If the chat history exceeds this ratio, the oldest messages will be flushed into long-term memory (if long-term memory is enabled).
- `token_flush_size` (default: 3000): The number of tokens to flush into long-term memory when the chat history exceeds the token limit.

```python
memory = Memory.from_defaults(
    session_id="my_session",
    token_limit=40000,
    chat_history_token_ratio=0.7,
    token_flush_size=3000,
)
```

### Long-Term Memory

Long-term memory is represented as `Memory Block` objects. These objects receive the messages that are flushed from short-term memory, and optionally process them to extract information. Then when memory is retrieved, the short-term and long-term memories are merged together.

Currently, there are three predefined memory blocks:

- `StaticMemoryBlock`: A memory block that stores a static piece of information.
- `FactExtractionMemoryBlock`: A memory block that extracts facts from the chat history.
- `VectorMemoryBlock`: A memory block that stores and retrieves batches of chat messages from a vector database.

By default, depending on the `insert_method` argument, the memory blocks will be inserted into the system message or the latest user message.

This sounds a bit complicated, but it's actually quite simple. Let's look at an example:

```python
from llama_index.core.memory import (
    StaticMemoryBlock,
    FactExtractionMemoryBlock,
    VectorMemoryBlock,
)

blocks = [
    StaticMemoryBlock(
        name="core_info",
        static_content="My name is Logan, and I live in Saskatoon. I work at LlamaIndex.",
        priority=0,
    ),
    FactExtractionMemoryBlock(
        name="extracted_info",
        llm=llm,
        max_facts=50,
        priority=1,
    ),
    VectorMemoryBlock(
        name="vector_memory",
        # required: pass in a vector store like qdrant, chroma, weaviate, milvus, etc.
        vector_store=vector_store,
        priority=2,
        embed_model=embed_model,
        # The top-k message batches to retrieve
        # similarity_top_k=2,
        # optional: How many previous messages to include in the retrieval query
        # retrieval_context_window=5
        # optional: pass optional node-postprocessors for things like similarity threshold, etc.
        # node_postprocessors=[...],
    ),
]
```

Here, we've setup three memory blocks:

- `core_info`: A static memory block that stores some core information about the user. The static content can either be a string or a list of `ContentBlock` objects like `TextBlock`, `ImageBlock`, etc. This information will always be inserted into the memory.
- `extracted_info`: An extracted memory block that will extract information from the chat history. Here we've passed in the `llm` to use to extarct facts from the flushed chat history, and set the `max_facts` to 50. If the number of extracted facts exceeds this limit, the `max_facts` will be automatically summarized and reduced to leave room for new information.
- `vector_memory`: A vector memory block that will store and retrieve batches of chat messages from a vector database. Each batch is a list of the flushed chat messages. Here we've passed in the `vector_store` and `embed_model` to use to store and retrieve the chat messages.

You'll also notice that we've set the `priority` for each block. This is used to determine the handling when the memory blocks content (i.e. long-term memory) + short-term memory exceeds the token limit on the `Memory` object.

When memory blocks get too long, they are automatically "truncated". By default, this just means they are removed from memory until there is room again. This can be customized with subclasses of memory blocks that implement their own truncation logic.

- `priority=0`: This block will always be kept in memory.
- `priority=1, 2, 3, etc`: This determines the order in which memory blocks are truncated when the memory exceeds the token limit, to help the overall short-term memory + long-term memory content be less than or equal to the `token_limit`.

Now, let's pass these blocks into the `Memory` class:

```python
memory = Memory.from_defaults(
    session_id="my_session",
    token_limit=40000,
    memory_blocks=blocks,
    insert_method="system",
)
```

As the memory is used, the short-term memory will fill up. Once the short-term memory exceeds the `chat_history_token_ratio`, the oldest messages that fit into the `token_flush_size` will be flushed and sent to each memory block for processing.

When memory is retrieved, the short-term and long-term memories are merged together. The `Memory` object will ensure that the short-term memory + long-term memory content is less than or equal to the `token_limit`. If it is longer, the `.truncate()` method will be called on the memory blocks, using the `priority` to determine the truncation order.

!!! tip
    By default, tokens are counted using tiktoken. To customize this, you can set the `tokenizer_fn` argument to a custom callable that given a string, returns a list. The length of the list is then used to determine the token count.

Once the memory has collected enough information, we might see something like this from the memory:

```python
# optionally pass in a list of messages to get, which will be forwarded to the memory blocks
chat_history = memory.get(messages=[...])

print(chat_history[0].content)
```

Which will print something like this:

```
<memory>
<static_memory>
My name is Logan, and I live in Saskatoon. I work at LlamaIndex.
</static_memory>
<fact_extraction_memory>
<fact>Fact 1</fact>
<fact>Fact 2</fact>
<fact>Fact 3</fact>
</fact_extraction_memory>
<retrieval_based_memory>
<message role='user'>Msg 1</message>
<message role='assistant'>Msg 2</message>
<message role='user'>Msg 3</message>
</retrieval_based_memory>
</memory>
```

Here, the memory was inserted into the system message, with specific sections for each memory block.

## Customizing Memory Blocks

While predefined memory blocks are available, you can also create your own custom memory blocks.

```python
from typing import Optional, List, Any
from llama_index.core.llms import ChatMessage
from llama_index.core.memory.memory import BaseMemoryBlock


# use generics to define the output type of the memory block
# can be str or List[ContentBlock]
class MentionCounter(BaseMemoryBlock[str]):
    """
    A memory block that counts the number of times a user mentions a specific name.
    """

    mention_name: str = "Logan"
    mention_count: int = 0

    async def _aget(
        self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any
    ) -> str:
        return f"Logan was mentioned {self.mention_count} times."

    async def _aput(self, messages: List[ChatMessage]) -> None:
        for message in messages:
            if self.mention_name in message.content:
                self.mention_count += 1

    async def atruncate(
        self, content: str, tokens_to_truncate: int
    ) -> Optional[str]:
        return ""
```

Here, we've defined a memory block that counts the number of times a user mentions a specific name.

Its truncate method is basic, just returning an empty string.

### Remote Memory

By default, the `Memory` class is using an in-memory SQLite database. You can plug in any remote database by changing the database URI.

You can customize the table name, and also optionally pass in an async engine directly. This is useful for managing your own connection pool.

```python
from llama_index.core.memory import Memory

memory = Memory.from_defaults(
    session_id="my_session",
    token_limit=40000,
    async_database_uri="postgresql+asyncpg://postgres:mark90@localhost:5432/postgres",
    # Optional: specify a table name
    # table_name="memory_table",
    # Optional: pass in an async engine directly
    # this is useful for managing your own connection pool
    # async_engine=engine,
)
```

## Memory vs. Workflow Context

At this point in the documentation, you may have encountered cases where you are using a Workflow and are serializing a `Context` object to save and resume a specific workflow state. The workflow `Context` is a complex object that holds runtime information about the workflow, as well as key/value pairs that are shared across workflow steps.

In comparison, the `Memory` object is a simpler object, holding only `ChatMessage` objects, and optionally a list of `MemoryBlock` objects for long-term memory.

In most practical cases, you will end up using both. If you aren't customizing the memory, then serializing the `Context` object will be sufficient.

```python
from llama_index.core.workflow import Context

ctx = Context(workflow)

# serialize the context
ctx_dict = ctx.to_dict()

# deserialize the context
ctx = Context.from_dict(workflow, ctx_dict)
```

In other cases, like when using `FunctionAgent`, `AgentWorkflow`, or `ReActAgent`, if you customize the memory, then you will want to provide that as a separate runtime argument (especially since beyond the default, the `Memory` object is not serializable).

```python
response = await agent.run("Hello!", memory=memory)
```

Lastly, there are cases ([like human-in-the-loop](../../../understanding/agent/human_in_the_loop.md)) where you will need to provide both the `Context` (to resume the workflow) and the `Memory` (to store the chat history).

```python
response = await agent.run("Hello!", ctx=ctx, memory=memory)
```

## (Deprecated) Memory Types

In `llama_index.core.memory`, we offer a few different memory types:

- `ChatMemoryBuffer`: A basic memory buffer that stores the last X messages that fit into a token limit.
- `ChatSummaryMemoryBuffer`: A memory buffer that stores the last X messages that fit into a token limit, and also summarizes the conversation periodically when it gets too long.
- `VectorMemory`: A memory that stores and retrieves chat messages from a vector database. It makes no guarantees about the order of the messages, and returns the most similar messages to the latest user message.
- `SimpleComposableMemory`: A memory that composes multiple memories together. Usually used to combine `VectorMemory` with `ChatMemoryBuffer` or `ChatSummaryMemoryBuffer`.

## Examples

You can find a few examples of memory in action below:

- [Memory](../../../examples/memory/memory.ipynb)
- [Manipulating Memory at Runtime](../../../examples/memory/custom_memory.ipynb)
- [Limiting Multi-Turn Confusion with Custom Memory](../../../examples/memory/custom_multi_turn_memory.ipynb)

**NOTE:** Deprecated examples:
- [Chat Memory Buffer](../../../examples/agent/memory/chat_memory_buffer.ipynb)
- [Chat Summary Memory Buffer](../../../examples/agent/memory/summary_memory_buffer.ipynb)
- [Composable Memory](../../../examples/agent/memory/composable_memory.ipynb)
- [Vector Memory](../../../examples/agent/memory/vector_memory.ipynb)
- [Mem0 Memory](../../../examples/memory/Mem0Memory.ipynb)
