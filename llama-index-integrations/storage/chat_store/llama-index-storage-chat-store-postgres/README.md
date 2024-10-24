# LlamaIndex Chat_Store Integration: Postgres Chat Store

## Installation

`pip install llama-index-storage-chat-store-postgres`

## Usage

Using `PostgresChatStore`, you can store your chat history remotely, without having to worry about manually persisting and loading the chat history.

```python
from llama_index.storage.chat_store.postgres import PostgresChatStore
from llama_index.core.memory import ChatMemoryBuffer

chat_store = PostgresChatStore.from_uri(
    uri="postgresql+asyncpg://postgres:password@127.0.0.1:5432/database",
)

chat_memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="user1",
)
```
