# Evaluating

Evaluation and benchmarking are crucial concepts in LLM development. To improve the performance of an LLM app (RAG, agents), you must have a way to measure it.

LlamaIndex offers key modules to measure the quality of generated results. We also offer key modules to measure retrieval quality. You can learn more about how evaluation works in LlamaIndex in our [module guides](/module_guides/evaluating/root.md).

## Response Evaluation

Does the response match the retrieved context? Does it also match the query? Does it match the reference answer or guidelines? Here's a simple example that evaluates a single response for Faithfulness, i.e. whether the response is aligned to the context, such as being free from hallucinations:

```python
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
from llama_index.evaluation import FaithfulnessEvaluator

# build service context
llm = OpenAI(model="gpt-4", temperature=0.0)
service_context = ServiceContext.from_defaults(llm=llm)

# build index
...
vector_index = VectorStoreIndex(...)

# define evaluator
evaluator = FaithfulnessEvaluator(service_context=service_context)

# query index
query_engine = vector_index.as_query_engine()
response = query_engine.query(
    "What battles took place in New York City in the American Revolution?"
)
eval_result = evaluator.evaluate_response(response=response)
print(str(eval_result.passing))
```

The response contains both the response and the source from which the response was generated; the evaluator compares them and determines if the response is faithful to the source.

You can learn more in our module guides about [response evaluation](/module_guides/evaluating/usage_pattern.md).

## Retrieval Evaluation

Are the retrieved sources relevant to the query? This is a simple example that evaluates a single retrieval:

```python
from llama_index.evaluation import RetrieverEvaluator

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

This compares what was retrieved for the query to a set of nodes that were expected to be retrieved.

In reality you would want to evaluate a whole batch of retrievals; you can learn how do this in our module guide on [retrieval evaluation](/module_guides/evaluating/usage_pattern_retrieval.md).

## Related concepts

You may be interested in [analyzing the cost of your application](/understanding/evaluating/cost_analysis/root.md) if you are making calls to a hosted, remote LLM.

```{toctree}
---
maxdepth: 1
hidden: true
---
/understanding/evaluating/cost_analysis/root.md
```
