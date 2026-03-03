# LlamaIndex Chat Store Integration: OpenSearch

## Installation

```
pip install llama-index-storage-chat-store-opensearch
```

## Usage

Using `OpensearchChatStore`, you can store your chat history in OpenSearch, without having to worry about manually persisting and loading the chat history.

```python
from llama_index.storage.chat_store.opensearch import OpensearchChatStore
from llama_index.core.memory import ChatMemoryBuffer

chat_store = OpensearchChatStore(
    opensearch_url="https://localhost:9200",
)

chat_memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="user1",
)
```

### Custom Index Name

```python
chat_store = OpensearchChatStore(
    opensearch_url="https://localhost:9200",
    index="my_chat_store",
)
```

### Using a Pre-configured Client

```python
from opensearchpy import OpenSearch

client = OpenSearch(
    "https://localhost:9200",
    http_auth=("admin", "admin"),
    verify_certs=False,
)

chat_store = OpensearchChatStore(
    opensearch_url="https://localhost:9200",
    os_client=client,
)
```
