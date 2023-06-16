# 🗃️ Data Index

## Concept
An `Index` is a data structure that allows us to quickly retrieve relevant context for a user query.
For LlamaIndex, it's the core foundation for retrieval-augmented generation (RAG) use-cases.


At a high-level, `Indices` are built from [Documents](/how_to/connector/root.md).
They are used to build [Query Engines](/how_to/query_engine/root.md) and [Chat Engines](/how_to/chat_engine/root.md)
which enables question & answer and chat over your data.  

Under the hood, `Indices` store data in `Node` objects (which represent chunks of the original documents), and expose an [Retriever](/how_to/retriever/root.md) interface that supports additional configuration and automation.


## Usage Pattern
Get started with:
```python
from llama_index import VectorStoreIndex

index = VectorStoreIndex.from_documents(docs)
```

```{toctree}
---
maxdepth: 2
---
usage_pattern.md
```


## Modules

```{toctree}
---
maxdepth: 2
---
modules.md
```

## Advanced Concepts

```{toctree}
---
maxdepth: 1
---
composability.md
```
