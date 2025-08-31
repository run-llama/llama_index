# LlamaIndex Chat_Store Integration: YugabyteDB Chat Store

## Installation

`pip install llama-index-storage-chat-store-yugabytedb`

## Usage

Using `YugabyteDBChatStore`, you can store your chat history remotely, without having to worry about manually persisting and loading the chat history.

```python
from llama_index.storage.chat_store.yugabytedb import YugabyteDBChatStore
from llama_index.core.memory import ChatMemoryBuffer

chat_store = YugabyteDBChatStore.from_uri(
    uri="postgresql+psycopg2://postgres:password@127.0.0.1:5433/yugabyte?load_balance=true",
)

chat_memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="user1",
)
```
