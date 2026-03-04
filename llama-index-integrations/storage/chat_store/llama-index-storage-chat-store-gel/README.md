# LlamaIndex Chat_Store Integration: Gel Chat Store

## Installation

`pip install llama-index-storage-chat-store-gel`

## Usage

Using `GelChatStore`, you can persist your chat history automatically and not have to worry about saving and loading it manually.

```python
from llama_index.storage.chat_store.gel import GelChatStore
from llama_index.core.memory import ChatMemoryBuffer

chat_store = GelChatStore()

chat_memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="user1",
)
```
