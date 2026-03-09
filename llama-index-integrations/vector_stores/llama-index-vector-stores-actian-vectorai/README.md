# LlamaIndex Vector Store Integration: Actian Vector AI

Actian Vector AI integration package for LlamaIndex using the `actian-vectorai` Python SDK.

## Installation

```bash
pip install llama-index-vector-stores-actian-vectorai
```

## Requirements

- Python `>=3.10,<3.13`
- A reachable Actian Vector AI endpoint (for example `localhost:50051`)
- A connected `VectorAIClient`
- An existing collection in Actian Vector AI

## Quick Start

```python
from actian_vectorai import VectorAIClient, VectorParams, Distance
from llama_index.vector_stores.actian_vectorai import ActianVectorAIVectorStore

COLLECTION_NAME = "llama_index_collection"

client = VectorAIClient("localhost:50051")
client.connect()

if not client.collections.exists(COLLECTION_NAME):
    client.collections.create(
        COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.Cosine),
    )

vector_store = ActianVectorAIVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
)
```

## Current Status

This integration is currently scaffolded.

- `add()` raises `NotImplementedError`
- `delete()` raises `NotImplementedError`
- `query()` raises `NotImplementedError`

## How Initialization Works

`ActianVectorAIVectorStore` validates two things at construction time:

1. `client.is_connected` must be `True`
2. `collection_name` must already exist in Actian Vector AI

If either condition is not met, the constructor raises `ValueError`.

## API

- Class: `ActianVectorAIVectorStore`
- Import path: `llama_index.vector_stores.actian_vectorai`
- `class_name() -> "ActianVectorAIVectorStore"`
- `client` property returns the underlying `VectorAIClient`

## Development

Run tests from this package directory:

```bash
uv run pytest -s
```
