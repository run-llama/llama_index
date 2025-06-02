# LlamaIndex Indices Integration: Bge-M3

## Setup

```
pip install -U llama-index-indices-managed-bge-m3
```

## Usage

```python
from llama_index.indices.managed.bge_m3 import BGEM3Index

index = BGEM3Index.from_documents(
    documents, weights_for_different_modes=[0.4, 0.2, 0.4]
)
retriever = index.as_retriever()
```
