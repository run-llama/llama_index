# LlamaIndex Storage_Chat_Store Integration: CockroachDB

A chat store backed by [CockroachDB](https://www.cockroachlabs.com/).

Each session's messages are stored as a single `JSONB` column holding a JSON array (CockroachDB does not yet support `ARRAY(JSON)` as a column type, see [crdb#23468](https://github.com/cockroachdb/cockroach/issues/23468)). Appends use the JSONB `||` operator inside an `INSERT ... ON CONFLICT DO UPDATE` so they are atomic and retry-safe under `SERIALIZATION_FAILURE`.

## Installation

```bash
pip install llama-index-storage-chat-store-cockroachdb
```

## Usage

```python
from llama_index.storage.chat_store.cockroachdb import CockroachDBChatStore
from llama_index.core.memory import ChatMemoryBuffer

chat_store = CockroachDBChatStore.from_params(
    host="localhost",
    port=26257,
    database="defaultdb",
    user="root",
    password="",
    sslmode="disable",  # local insecure cluster only
    table_name="chat_history",
)

memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="user1",
)
```
