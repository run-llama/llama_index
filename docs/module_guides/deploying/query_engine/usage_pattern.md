# Usage Pattern

## Get Started

Build a query engine from index:

```python
query_engine = index.as_query_engine()
```

```{tip}
To learn how to build an index, see [Indexing](/module_guides/indexing/indexing.md)
```

Ask a question over your data

```python
response = query_engine.query("Who is Paul Graham?")
```

## Configuring a Query Engine

### High-Level API

You can directly build and configure a query engine from an index in 1 line of code:

```python
query_engine = index.as_query_engine(
    response_mode="tree_summarize",
    verbose=True,
)
```

> Note: While the high-level API optimizes for ease-of-use, it does _NOT_ expose full range of configurability.

See [**Response Modes**](./response_modes.md) for a full list of response modes and what they do.

```{toctree}
---
maxdepth: 1
hidden:
---
response_modes.md
streaming.md
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

### Streaming

To enable streaming, you simply need to pass in a `streaming=True` flag

```python
query_engine = index.as_query_engine(
    streaming=True,
)
streaming_response = query_engine.query(
    "What did the author do growing up?",
)
streaming_response.print_response_stream()
```

- Read the full [streaming guide](/module_guides/deploying/query_engine/streaming.md)
- See an [end-to-end example](/examples/customization/streaming/SimpleIndexDemo-streaming.ipynb)

## Defining a Custom Query Engine

You can also define a custom query engine. Simply subclass the `CustomQueryEngine` class, define any attributes you'd want to have (similar to defining a Pydantic class), and implement a `custom_query` function that returns either a `Response` object or a string.

```python
from llama_index.query_engine import CustomQueryEngine
from llama_index.retrievers import BaseRetriever
from llama_index.response_synthesizers import (
    get_response_synthesizer,
    BaseSynthesizer,
)


class RAGQueryEngine(CustomQueryEngine):
    """RAG Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        response_obj = self.response_synthesizer.synthesize(query_str, nodes)
        return response_obj
```

See the [Custom Query Engine guide](/examples/query_engine/custom_query_engine.ipynb) for more details.
