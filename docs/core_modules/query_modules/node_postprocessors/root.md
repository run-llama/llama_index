# Node Postprocessor

## Concept
Node postprocessors are a set of modules that take a set of nodes, and apply some kind of transformation or filtering before returning them.

In LlamaIndex, node postprocessors are most commonly applied within a query engine, after the node retrieval step and before the response synthesis step.

LlamaIndex offers several node postprocessors for immediate use, while also providing a simple API for adding your own custom postprocessors.

## Usage Pattern

An example of using a node postprocessors is below:

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

You can find more details using post processors and how to build your own below.

```{toctree}
---
maxdepth: 2
---
usage_pattern.md
```

## Modules
Below you can find guides for each node postprocessor.

```{toctree}
---
maxdepth: 2
---
modules.md
```