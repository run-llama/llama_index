---
title: Usage Pattern (Retrieval)
---

## Using `RetrieverEvaluator`

This runs evaluation over a single query + ground-truth document set given a retriever.

The standard practice is to specify a set of valid metrics with `from_metrics`.

```python
from llama_index.core.evaluation import RetrieverEvaluator

# define retriever somewhere (e.g. from index)
# retriever = index.as_retriever(similarity_top_k=2)
retriever = ...

retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever
)

retriever_evaluator.evaluate(
    query="query", expected_ids=["node_id1", "node_id2"]
)
```

## Breaking results down by segment

`SegmentedRetrieverEvaluator` wraps a `RetrieverEvaluator` and additionally computes every metric per segment, e.g. per source format, language or document collection. `segment_fn` labels each retrieved node, and `expected_segments` labels the expected ids, usually built once from the nodes the index was built from.

```python
from llama_index.core.evaluation import SegmentedRetrieverEvaluator

segmented_evaluator = SegmentedRetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate", "recall"],
    base_evaluator=retriever_evaluator,
    segment_fn=lambda node: node.metadata["format"],
    segments=["table", "prose"],
    expected_segments={
        node.node_id: node.metadata["format"] for node in nodes
    },
)

result = segmented_evaluator.evaluate(
    query="query", expected_ids=["node_id1", "node_id2"]
)
result.segment_metric_vals_dict  # metrics per segment
result.not_applicable_segments  # segments without relevant expected ids
```

Each segment is scored against its own expected ids. A segment with relevant expected ids but no retrieved nodes scores 0.0, while a segment without any relevant expected ids is listed in `not_applicable_segments` instead of being scored, so it does not distort averages across queries. Ranks are counted within each segment, not across the full result list.

## Building an Evaluation Dataset

You can manually curate a retrieval evaluation dataset of questions + node id's. We also offer synthetic dataset generation over an existing text corpus with our `generate_question_context_pairs` function:

```python
from llama_index.core.evaluation import generate_question_context_pairs

qa_dataset = generate_question_context_pairs(
    nodes, llm=llm, num_questions_per_chunk=2
)
```

The returned result is a `EmbeddingQAFinetuneDataset` object (containing `queries`, `relevant_docs`, and `corpus`).

### Plugging it into `RetrieverEvaluator`

We offer a convenience function to run a `RetrieverEvaluator` over a dataset in batch mode.

```python
eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
```

This should run much faster than you trying to call `.evaluate` on each query separately.
