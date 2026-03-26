# LlamaIndex Chat_Store Integration: YugabyteDB Chat Store

## Prerequisites

Before using this integration, you'll need to have a YugabyteDB instance running. You can set up a local YugabyteDB instance by following the [YugaByteDB Quick Start Guide](https://docs.yugabyte.com/preview/quick-start/macos/).

## Installation

```shell
pip install llama-index-storage-chat-store-yugabytedb
```

## Usage

Using `YugabyteDBChatStore`, you can store your chat history remotely, without having to worry about manually persisting and loading the chat history.

```python
from llama_index.storage.chat_store.yugabytedb import YugabyteDBChatStore
from llama_index.core.memory import ChatMemoryBuffer

chat_store = YugabyteDBChatStore.from_uri(
    uri="yugabytedb+psycopg2://yugabyte:password@127.0.0.1:5433/yugabyte?load_balance=true",
)

chat_memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="user1",
)
```

### Connection String Parameters

The connection string passed to `YugabyteDBChatStore.from_uri()` supports various parameters that can be used to configure the connection to your YugabyteDB cluster.
You can find a complete list of supported parameters in the [YugabyteDB psycopg2 Driver Documentation](https://docs.yugabyte.com/preview/drivers-orms/python/yugabyte-psycopg2/#step-2-set-up-the-database-connection).

The YugabyteDB specific parameters include:

- `load_balance`: Enable/disable load balancing (default: false)
- `topology_keys`: Specify preferred nodes for connection routing
- `yb_servers_refresh_interval`: Interval (in seconds) to refresh the list of available servers
- `fallback_to_topology_keys_only`: Whether to only connect to nodes specified in topology_keys
- `failed_host_ttl_seconds`: Time (in seconds) to wait before trying to connect to failed nodes
