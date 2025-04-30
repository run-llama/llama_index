# Contextual Reranker

This is a Llama_index package that calls Contextual's `/rerank` endpoint. It will rank a list of documents according to their relevance to a query.

The total request cannot exceed 400,000 tokens. The combined length of any document, instruction and the query must not exceed 4,000 tokens. Email [rerank-feedback@contextual.ai](mailto:rerank-feedback@contextual.ai) with any feedback or questions.

## Usage

```python
from llama_index.postprocessor.contextual_rerank import ContextualRerank
from llama_index.core.schema import NodeWithScore, TextNode

nodes = [
    NodeWithScore(node=TextNode(text="the capital of france is paris")),
    NodeWithScore(
        node=TextNode(text="the capital of the United States is Washington DC")
    ),
]

query = "What is the capital of France?"

contextual_rerank = ContextualRerank(
    api_key="key-...",
    model="ctxl-rerank-en-v1-instruct",
    top_n=2,
)

response = contextual_rerank.postprocess_nodes(nodes, query_str=query)

for node in response:
    print(node)
```
