# Usage Pattern

## Get Started
Get a retriever from index:
```python
retriever = index.as_retriever()
```

Retrieve relevant context for a question:
```python
nodes = retriever.retrieve('Who is Paul Graham?')
```

> Note: To learn how to build an index, see [Index](/core_modules/data_modules/index/root.md)

## High-Level API

### Selecting a Retriever

You can select the index-specific retriever class via `retriever_mode`. 
For example, with a `ListIndex`:
```python
retriever = list_index.as_retriever(
    retriever_mode='llm',
)
```
This creates a [ListIndexLLMRetriever](/api_reference/query/retrievers/list.rst) on top of the list index.

See [**Retriever Modes**](/core_modules/query_modules/retriever/retriever_modes.md) for a full list of (index-specific) retriever modes
and the retriever classes they map to.

```{toctree}
---
maxdepth: 1
hidden:
---
retriever_modes.md
```

### Configuring a Retriever
In the same way, you can pass kwargs to configure the selected retriever.
> Note: take a look at the API reference for the selected retriever class' constructor parameters for a list of valid kwargs.

For example, if we selected the "llm" retriever mode, we might do the following:
```python
retriever = list_index.as_retriever(
    retriever_mode='llm',
    choice_batch_size=5,
)

```

## Low-Level Composition API
You can use the low-level composition API if you need more granular control.  

To achieve the same outcome as above, you can directly import and construct the desired retriever class:
```python
from llama_index.indices.list import ListIndexLLMRetriever

retriever = ListIndexLLMRetriever(
    index=list_index,
    choice_batch_size=5,
)
```


## Advanced

```{toctree}
---
maxdepth: 1
---
Define Custom Retriever </examples/query_engine/CustomRetrievers.ipynb>
BM25 Hybrid Retriever </examples/retrievers/bm25_retriever.ipynb>
```