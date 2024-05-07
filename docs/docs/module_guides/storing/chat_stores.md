# Chat Stores

A chat store serves as a centralized interface to store your chat history. Chat history is unique compared to other storage formats, since the order of messages is important for maintaining an overall conversation.

Chat stores can organize sequences of chat messages by keys (like `user_ids` or other unique identifiable strings), and handle `delete`, `insert`, and `get` operations.

## SimpleChatStore

The most basic chat store is `SimpleChatStore`, which stores messages in memory and can save to/from disk, or can be serialized and stored elsewhere.

Typically, you will instantiate a chat store and give it to a memory module. Memory modules that use chat stores will default to using `SimpleChatStore` if not provided.

```python
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer

chat_store = SimpleChatStore()

chat_memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="user1",
)
```

Once you have the memory created, you might include it in an agent or chat engine:

```python
agent = OpenAIAgent.from_tools(tools, memory=memory)
# OR
chat_engine = index.as_chat_engine(memory=memory)
```

To save the chat store for later, you can either save/load from disk

```python
chat_store.persist(persist_path="chat_store.json")
loaded_chat_store = SimpleChatStore.from_persist_path(
    persist_path="chat_store.json"
)
```

Or you can convert to/from a string, saving the string somewhere else along the way

```python
chat_store_string = chat_store.json()
loaded_chat_store = SimpleChatStore.parse_raw(chat_store_string)
```

## RedisChatStore

Using `RedisChatStore`, you can store your chat history remotely, without having to worry about manually persisting and loading the chat history.

```python
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.core.memory import ChatMemoryBuffer

chat_store = RedisChatStore(redis_url="redis://localhost:6379", ttl=300)

chat_memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="user1",
)
```
