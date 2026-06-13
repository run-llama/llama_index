# LlamaIndex Memory Integration: Recollect

[Recollect](https://github.com/cobusgreyling/recollect) is a self-hosted long-term memory layer for AI agents. This package provides a LlamaIndex `BaseMemory` adapter.

## Installation

```bash
pip install llama-index-memory-recollect
```

## Quickstart (local, no API keys)

```python
from llama_index.memory.recollect import RecollectMemory
from recollect.config import RecollectConfig

memory = RecollectMemory.from_config(
    context={"user_id": "alice"},
    config=RecollectConfig.local_dev().model_dump(),
    search_msg_limit=4,
)
```

Use with `SimpleChatEngine`, `FunctionAgent`, or `ReActAgent` by passing `memory=memory`.

## Notebook

See [`docs/examples/memory/RecollectMemory.ipynb`](../../../docs/examples/memory/RecollectMemory.ipynb).

## Links

- [Recollect GitHub](https://github.com/cobusgreyling/recollect)
- [LlamaHub memory integrations](https://llamahub.ai/l/memory)