# LlamaIndex Vector Store Integration: VDE

VDE integration package for LlamaIndex using the `actiancortex` (`cortex`) Python SDK.

## Installation

```bash
pip install llama-index-vector-stores-vde
```

## Requirements

- Python `>=3.10`
- A reachable Cortex/VDE endpoint (for example `localhost:50051`)
- Optional API key if your server requires authentication

## Quick Start

```python
from cortex import DistanceMetric
from llama_index.vector_stores.vde import VDEVectorStore

vector_store = VDEVectorStore(
    address="localhost:50051",
    api_key=None,
    collection_name="llama_index_collection",
    collection_dimension=1536,
    distance_metric=DistanceMetric.COSINE,
)
```

## Current Status

This package is currently a scaffold.

- `add()` is not implemented.
- `delete()` is not implemented.
- `query()` is not implemented.

Each of these methods currently raises `NotImplementedError`.

## Constructor Parameters

- `address`: Cortex server address (`host:port`)
- `api_key`: API key for authenticated deployments
- `pool_size`: gRPC channel pool size
- `enable_smart_batching`: Enable SDK smart batching
- `batch_size`: Smart batching max items
- `batch_timeout_ms`: Smart batching max wait time
- `timeout`: Request timeout in seconds
- `collection_name`: Collection to open/create
- `collection_dimension`: Vector dimension
- `distance_metric`: `DistanceMetric` (`COSINE`, `EUCLIDEAN`, `DOT`)
- `hnsw_m`, `hnsw_ef_construct`, `hnsw_ef_search`: HNSW index params
- `config_json`: Optional backend-specific config

## Development

Run tests from this package directory:

```bash
uv run pytest
```
