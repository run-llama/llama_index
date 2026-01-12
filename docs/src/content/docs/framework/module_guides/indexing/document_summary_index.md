---
title: Document Summary Index
---

The Document Summary Index is a data structure that stores a summary for each document and maps that summary to the underlying nodes. This allows for efficient retrieval based on document-level context rather than just individual node context.

## Concept

Unlike a standard `VectorStoreIndex` which embeds individual nodes, the `DocumentSummaryIndex` generates a summary for each document (using an LLM) and embeds those summaries.

1. **Summarization**: For each document, the index uses an LLM to generate a summary.
2. **Mapping**: The summary is mapped to all nodes belonging to that document.
3. **Retrieval**: During query time, the index can retrieve the most relevant summaries and then fetch the corresponding source nodes.

## Usage Pattern

### Building the Index

```python
from llama_index.core import DocumentSummaryIndex, SimpleDirectoryReader, StorageContext
from llama_index.core import Settings

# Load documents
documents = SimpleDirectoryReader("./data").load_data()

# Create the index
index = DocumentSummaryIndex.from_documents(
    documents,
    show_progress=True,
)
```

### Persistence and Database Backends

When using database-backed storage (such as PostgreSQL/pgvector, AlloyDB, or ChromaDB), it is crucial to ensure that both the nodes and their summaries are correctly persisted in the `docstore` and `indexstore`.

A common issue (#19605) arises when using a shared `StorageContext` where nodes might not be explicitly stored, leading to `KeyError` during retrieval.

To ensure persistence, you can use the `store_nodes_override` parameter:

```python
index = DocumentSummaryIndex.from_documents(
    documents,
    storage_context=storage_context,
    store_nodes_override=True
)
```

### Retrieval

The `DocumentSummaryIndex` supports two main retrieval modes:

1. **Embedding-based**: Retrieves documents by comparing the query embedding with the document summary embeddings.
2. **LLM-based**: Uses an LLM to select the most relevant documents based on their summaries.

```python
# Embedding-based retrieval (default)
retriever = index.as_retriever(retriever_mode="embedding")

# LLM-based retrieval
retriever = index.as_retriever(retriever_mode="llm")
```

## API Reference

See the [API Reference](/python/api_reference/indices/document_summary) for more details.
