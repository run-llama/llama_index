# Usage Pattern

## Get Started
Build a query engine from index:
```python
query_engine = index.as_query_engine()
```

> Note: To learn how to build an index, see [Index](/how_to/index/root.md)

Ask a question over your data
```python
response = query_engine.query('Who is Paul Graham?')
```

## Configuring a Query Engine
### High-Level API
You can directly build and configure a query engine from an index in 1 line of code:
```python
query_engine = index.as_query_engine(
    response_mode='tree_summarize',
    verbose=True,
)
```
> Note: While the high-level API optimizes for ease-of-use, it does *NOT* expose full range of configurability.  

See [**Response Modes**](/how_to/query_engine/response_modes.md) for a full list of response modes and what they do.

```{toctree}
---
maxdepth: 1
hidden:
---
response_modes.md
```



### Low-Level Composition API

You can use the low-level composition API if you need more granular control.
Concretely speaking, you would explicitly construct a `QueryEngine` object instead of calling `index.as_query_engine(...)`.
> Note: You may need to look at API references or example notebooks.


```python
from llama_index import (
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine

# build index
index = VectorStoreIndex.from_documents(documents)

# configure retriever
retriever = VectorIndexRetriever(
    index=index, 
    similarity_top_k=2,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
)

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

# query
response = query_engine.query("What did the author do growing up?")
print(response)
```
