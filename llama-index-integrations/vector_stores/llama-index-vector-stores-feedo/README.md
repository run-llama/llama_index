# LlamaIndex Vector_Stores Integration: Feedo

This package contains the LlamaIndex integration for Feedo, a decentralized long-term memory network.
Feedo offers encrypted at rest decentralized memory storage for AI agents, bounded by customizable tenant boundaries (namespaces).

## Installation

```bash
pip install llama-index-vector-stores-feedo
```

## Usage

```python
from llama_index.vector_stores.feedo import FeedoVectorStore

# Initialize the vector store
vector_store = FeedoVectorStore(
    usage_key="your_feedo_usage_key",
    did="your_agent_did",
    namespace="your_tenant_id"
)
```
