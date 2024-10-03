# LlamaIndex Chat_Store Integration: Upstash Chat Store

## Installation

`pip install llama-index-storage-chat-store-upstash`

## Usage

```python
from llama_index.storage.chat_store.upstash import UpstashChatStore
from llama_index.core.memory import ChatMemoryBuffer

chat_store = UpstashChatStore(
    redis_url="YOUR_UPSTASH_REDIS_URL",
    redis_token="YOUR_UPSTASH_REDIS_TOKEN",
    ttl=300,  # Optional: Time to live in seconds
)

chat_memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="user1",
)
```

UpstashChatStore supports both synchronous and asynchronous operations. Here's an example of using async methods:

```python
import asyncio
from llama_index.core.llms import ChatMessage


async def main():
    # Add messages
    messages = [
        ChatMessage(content="Hello", role="user"),
        ChatMessage(content="Hi there!", role="assistant"),
    ]
    await chat_store.async_set_messages("conversation1", messages)

    # Retrieve messages
    retrieved_messages = await chat_store.async_get_messages("conversation1")
    print(retrieved_messages)

    # Delete last message
    deleted_message = await chat_store.async_delete_last_message(
        "conversation1"
    )
    print(f"Deleted message: {deleted_message}")


asyncio.run(main())
```
