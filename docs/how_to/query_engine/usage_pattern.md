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


### Low-Level Composition API

You can use the low-level composition API if you need more granular control.
Concretely speaking, you would explicitly construct a `QueryEngine` object instead of calling `index.as_query_engine(...)`.
> Note: You may need to look at API references or example notebooks.


```python
from llama_index import (
    GPTVectorStoreIndex,
    ResponseSynthesizer,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine

# build index
index = GPTVectorStoreIndex.from_documents(documents)

# configure retriever
retriever = VectorIndexRetriever(
    index=index, 
    similarity_top_k=2,
)

# configure response synthesizer
response_synthesizer = ResponseSynthesizer.from_args(
    response_mode="tree_summarize",
    verbose=True,
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

## Advanced Configurations
You can further configure the query engine with [advanced components](/how_to/query_engine/advanced/root.md)
to reduce token cost, improve retrieval quality, etc. 