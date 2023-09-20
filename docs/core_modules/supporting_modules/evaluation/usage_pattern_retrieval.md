# Usage Pattern (Retrieval)

## Using `RetrieverEvaluator`

This runs evaluation over a single query + ground-truth document set given a retriever.

The standard practice is to specify a set of valid metrics with `from_metrics`.

```python
from llama_index.evaluation import RetrieverEvaluator

# define retriever somewhere (e.g. from index)
# retriever = index.as_retriever(similarity_top_k=2)
retriever = ...

retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever
)

retriever_evaluator.evaluate(
    query="query",
    expected_ids=["node_id1", "node_id2"]
)
```
