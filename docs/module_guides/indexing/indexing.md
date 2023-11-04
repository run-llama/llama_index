# Indexing

## Concept

An `Index` is a data structure that allows us to quickly retrieve relevant context for a user query.
For LlamaIndex, it's the core foundation for retrieval-augmented generation (RAG) use-cases.

At a high-level, `Indexes` are built from [Documents](/module_guides/loading/documents_and_nodes/root.md).
They are used to build [Query Engines](/module_guides/deploying/query_engine/root.md) and [Chat Engines](/module_guides/deploying/chat_engines/root.md)
which enables question & answer and chat over your data.

Under the hood, `Indexes` store data in `Node` objects (which represent chunks of the original documents), and expose a [Retriever](/module_guides/querying/retriever/root.md) interface that supports additional configuration and automation.

For a more in-depth explanation, check out our guide below:

```{toctree}
---
maxdepth: 1
---
index_guide.md
```

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
