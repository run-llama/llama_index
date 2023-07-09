# Usage Pattern

Most commonly, node-postprocessors will be used in a query engine, where they are applied to the nodes returned from a retriever, and before the response synthesis step.


## Using with a Query Engine

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.indices.postprocessor import TimeWeightedPostprocessor

documents = SimpleDirectoryReader("./data").load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(
  node_postprocessors=[
    TimeWeightedPostprocessor(
        time_decay=0.5, time_access_refresh=False, top_k=1
    )
  ]
)

# all node post-processors will be applied during each query
response = query_engine.query("query string")
```

## Using with Retrieved Nodes

Or used as a standalone object for filtering retrieved nodes:

```python
from llama_index.indices.postprocessor import SimilarityPostprocessor

nodes = index.as_retriever().query("query string")

# filter nodes below 0.75 similarity score
processor = SimilarityPostprocessor(similarity_cutoff=0.75)
filtered_nodes = processor.postprocess_nodes(nodes)
```

## Using with your own nodes

As you may have noticed, the postprocessors take `NodeWithScore` objects as inputs, which is just a wrapper class with a `Node` and a `score` value.

```python
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.schema import Node, NodeWithScore

nodes = [
  NodeWithScore(node=Node(text="text"), score=0.7),
  NodeWithScore(node=Node(text="text"), score=0.8)
]

# filter nodes below 0.75 similarity score
processor = SimilarityPostprocessor(similarity_cutoff=0.75)
filtered_nodes = processor.postprocess_nodes(nodes)
```
