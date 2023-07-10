# Usage Pattern

## Get Started

Build an index from documents:

```python
from llama_index import VectorStoreIndex

index = VectorStoreIndex.from_documents(docs)
```

```{tip}
To learn how to load documents, see [Data Connectors](/core_modules/data_modules/connector/root.md)
```

### What is happening under the hood?

1. Documents are chunked up and parsed into `Node` objects (which are lightweight abstraction over text str that additional keep track of metadata and relationships).
2. Additional computation is performed to add `Node` into index data structure
   > Note: the computation is index-specific.
   >
   > - For a vector store index, this means calling an embedding model (via API or locally) to compute embedding for the `Node` objects
   > - For a document summary index, this means calling an LLM to generate a summary

## Configuring Document Parsing

The most common configuration you might want to change is how to parse document into `Node` objects.

### High-Level API

We can configure our service context to use the desired chunk size and set `show_progress` to display a progress bar during index construction.

```python
from llama_index import ServiceContext, VectorStoreIndex

service_context = ServiceContext.from_defaults(chunk_size=512)
index = VectorStoreIndex.from_documents(
    docs,
    service_context=service_context,
    show_progress=True
)
```

> Note: While the high-level API optimizes for ease-of-use, it does _NOT_ expose full range of configurability.

### Low-Level API

You can use the low-level composition API if you need more granular control.

Here we show an example where you want to both modify the text chunk size, disable injecting metadata, and disable creating `Node` relationships.  
The steps are:

1. Configure a node parser

```python
from llama_index.node_parser import SimpleNodeParser

parser = SimpleNodeParser.from_defaults(
    chunk_size=512,
    include_extra_info=False,
    include_prev_next_rel=False,
)
```

2. Parse document into `Node` objects

```python
nodes = parser.get_nodes_from_documents(documents)
```

3. build index from `Node` objects

```python
index = VectorStoreIndex(nodes)
```

## Handling Document Update

Read more about how to deal with data sources that change over time with `Index` **insertion**, **deletion**, **update**, and **refresh** operations.

```{toctree}
---
maxdepth: 1
---
metadata_extraction.md
document_management.md
```
