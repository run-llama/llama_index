# Second-Stage Processing

By default, when a query is executed on an index or a composed graph, 
LlamaIndex performs the following steps:
1. **Retrieval step**: Retrieve a set of nodes from the index given the query. 
2. **Synthesis step**: Synthesize a response over the set of nodes.

Beyond standard retrieval and synthesis, LlamaIndex also provides a collection of modules
for advanced **second-stage processing** (i.e. after retrieval and before synthesis).

After retrieving the initial candidate nodes, these modules further improve
the quality and diversity of the nodes used for synthesis by e.g. filtering, re-ranking, or augmenting.
Examples include keyword filters, LLM-based re-ranking, and temporal-reasoning based augmentation.


We first provide the high-level API interface, and provide some example modules, and finally discuss usage.

We are also very open to contributions! Take a look at our [contribution guide](https://github.com/jerryjliu/llama_index/blob/main/CONTRIBUTING.md) if you 
are interested in contributing a Postprocessor.

## API Interface

The base class is `BaseNodePostprocessor`, and the API interface is very simple: 

```python

class BaseNodePostprocessor:
    """Node postprocessor."""

    @abstractmethod
    def postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
```

It takes in a list of Node objects, and outputs another list of Node objects.

The full API reference can be found [here](/reference/node_postprocessor.rst).


## Example Usage

The postprocessor can be used as part of a `QueryEngine`, or on its own.

#### Index querying

```python

from llama_index.indices.postprocessor import (
    FixedRecencyPostprocessor,
)
node_postprocessor = FixedRecencyPostprocessor(service_context=service_context)

query_engine = index.as_query_engine(
    similarity_top_k=3,
    node_postprocessors=[node_postprocessor]
)
response = query_engine.query(
    "How much did the author raise in seed funding from Idelle's husband (Julian) for Viaweb?", 
)

```


#### Using as Independent Module (Lower-Level Usage)

The module can also be used on its own as part of a broader flow. For instance,
here's an example where you choose to manually postprocess an initial set of source nodes.

```python
from llama_index.indices.postprocessor import (
    FixedRecencyPostprocessor,
)

# get initial response from vector index
query_engine = index.as_query_engine(
    similarity_top_k=3,
    response_mode="no_text"
)
init_response = query_engine.query(query_str)
resp_nodes = [n.node for n in init_response.source_nodes]

# use node postprocessor to filter nodes
node_postprocessor = FixedRecencyPostprocessor(service_context=service_context)
new_nodes = node_postprocessor.postprocess_nodes(resp_nodes)

# use list index to synthesize answers
list_index = ListIndex(new_nodes)
query_engine = list_index.as_query_engine(
    node_postprocessors=[node_postprocessor]
)
response = query_engine.query(query_str)
```


## Example Modules

### Default Postprocessors

These postprocessors are simple modules that are already included by default.

#### KeywordNodePostprocessor

A simple postprocessor module where you are able to specify `required_keywords` or `exclude_keywords`.
This will filter out nodes that don't have required keywords, or contain excluded keywords.

#### SimilarityPostprocessor

The SimilarityPostprocessor module allows you to filter nodes based on their similarity to a reference node using cosine similarity. By setting a `similarity_cutoff`, you can define a minimum similarity score that nodes must exceed to be considered relevant. This helps retrieve nodes that are closely related to the reference node in terms of their semantic meaning. Leveraging cosine similarity as a default, this postprocessor enhances the retrieval process by fine-tuning the relevance of the retrieved nodes based on their similarity to the reference node.

For example, if you have a reference node and you want to retrieve nodes that are highly similar to it, you can set a high similarity cutoff. This will exclude nodes with low similarity scores and only retain nodes that have a similarity score above the specified cutoff.

#### Previous/Next Postprocessors

These postprocessors are able to exploit temporal relationships between nodes
(e.g. prev/next relationships) in order to retrieve additional
context, in the event that the existing context may not directly answer
the question. They augment the set of retrieved nodes with context
either in the future or the past (or both).

The most basic version is `PrevNextNodePostprocessor`, which takes a fixed
`num_nodes` as well as `mode` specifying "previous", "next", or "both".

We also have `AutoPrevNextNodePostprocessor`, which is able to infer
the `previous`, `next` direction.

![](/_static/node_postprocessors/prev_next.png)


#### Recency Postprocessors

These postprocessors are able to ensure that only the most recent
data is used as context, and that out of date context information is filtered out.

Imagine that you have three versions of a document, with slight changes between versions. For instance, this document may be describing patient history. If you ask a question over this data, you would want to make sure that you're referencing the latest document, and that out of date information is not passed in.

We support recency filtering through the following modules.

**`FixedRecencyPostProcessor`**: sorts retrieved nodes by date in reverse order, and takes a fixed top-k set of nodes.

![](/_static/node_postprocessors/recency.png)

**`EmbeddingRecencyPostprocessor`**: sorts retrieved nodes by date in reverse order, and then
looks at subsequent nodes and filters out nodes that have high embedding 
similarity with the current node. This allows us to maintain recent Nodes
that have "distinct" context, but filter out overlapping Nodes that
are outdated and overlap with more recent context.


**`TimeWeightedPostprocessor`**: adds time-weighting to retrieved nodes, using the formula `(1-time_decay) ** hours_passed`.
The recency score is added to any score that the node already contains.


# Token Optimizer PostProcessor

Our **`SentenceEmbeddingOptimizer`** is a postprocessor that will optimize for token usage. It does this by removing words and sentences that are not relevant to the query.

Here is a sample code snippet on comparing the outputs without optimization and with.

```python
from llama_index import VectorStoreIndex
from llama_index.indices.postprocessor import SentenceEmbeddingOptimizer
print("Without optimization")
start_time = time.time()
query_engine = index.as_query_engine()
res = query_engine.query("What is the population of Berlin?")
end_time = time.time()
print("Total time elapsed: {}".format(end_time - start_time))
print("Answer: {}".format(res))

print("With optimization")
start_time = time.time()
query_engine = index.as_query_engine(
    node_postprocessors=[SentenceEmbeddingOptimizer(percentile_cutoff=0.5)]
)
res = query_engine.query("What is the population of Berlin?")
end_time = time.time()
print("Total time elapsed: {}".format(end_time - start_time))
print("Answer: {}".format(res))

```

Output:
```text
Without optimization
INFO:root:> [query] Total LLM token usage: 3545 tokens
INFO:root:> [query] Total embedding token usage: 7 tokens
Total time elapsed: 2.8928110599517822
Answer: 
The population of Berlin in 1949 was approximately 2.2 million inhabitants. After the fall of the Berlin Wall in 1989, the population of Berlin increased to approximately 3.7 million inhabitants.

With optimization
INFO:root:> [optimize] Total embedding token usage: 7 tokens
INFO:root:> [query] Total LLM token usage: 1779 tokens
INFO:root:> [query] Total embedding token usage: 7 tokens
Total time elapsed: 2.346346139907837
Answer: 
The population of Berlin is around 4.5 million.
```

Full [example notebook here](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/node_postprocessor/OptimizerDemo.ipynb).

```{toctree}
---
caption: Examples
maxdepth: 1
---
/examples/node_postprocessor/PrevNextPostprocessorDemo.ipynb
/examples/node_postprocessor/RecencyPostprocessorDemo.ipynb
/examples/node_postprocessor/TimeWeightedPostprocessorDemo.ipynb
/examples/node_postprocessor/PII.ipynb
/examples/node_postprocessor/CohereRerank.ipynb
/examples/node_postprocessor/LLMReranker-Gatsby.ipynb
/examples/node_postprocessor/OptimizerDemo.ipynb
```