# LlamaIndex Chat_Store Integration: SuperLocalMemory Chat Store

A local-first, privacy-first chat store backed by SuperLocalMemory V2. All data stays on your machine in SQLite — no cloud, no API keys, no subscriptions.

## Installation

```bash
pip install llama-index-storage-chat-store-superlocalmemory
```

### Prerequisites

SuperLocalMemory V2 must be installed:

```bash
npm install -g superlocalmemory
```

## Usage

```python
from llama_index.storage.chat_store.superlocalmemory import SuperLocalMemoryChatStore
from llama_index.core.memory import ChatMemoryBuffer

chat_store = SuperLocalMemoryChatStore()

chat_memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="user1",
)
```

You can also specify a custom database path:

```python
chat_store = SuperLocalMemoryChatStore(
    db_path="/path/to/custom/memory.db",
)
```

SuperLocalMemory supports both synchronous and asynchronous operations:

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

## Features

- **100% local**: All data stored in SQLite on your machine (`~/.claude-memory/memory.db`)
- **Zero cloud dependencies**: No API keys, no external services, no subscriptions
- **All 7 BaseChatStore methods**: Full implementation of the LlamaIndex chat store interface
- **Session isolation**: Chat sessions are isolated via hashed tags
- **Universal memory**: Shared across 17+ AI tools (Claude, Cursor, Windsurf, and more)
- **Free and open-source**: MIT licensed

## Links

- **GitHub**: https://github.com/varun369/SuperLocalMemoryV2
- **PyPI**: https://pypi.org/project/llama-index-storage-chat-store-superlocalmemory/
- **Docs**: https://varun369.github.io/SuperLocalMemoryV2/
