# LlamaIndex Chat_Store Integration: Gel Chat Store

## Installation

`pip install llama-index-storage-chat-store-gel`

## Usage

Using `GelChatStore`, you can store your chat history remotely, without having to worry about manually persisting and loading the chat history.

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
