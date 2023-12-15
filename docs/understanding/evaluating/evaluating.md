# Evaluating

Evaluation and benchmarking are crucial concepts in LLM development. To improve the performance of an LLM app (RAG, agents), you must have a way to measure it.

LlamaIndex offers key modules to measure the quality of generated results. We also offer key modules to measure retrieval quality.

- **Response Evaluation**: Does the response match the retrieved context? Does it also match the query? Does it match the reference answer or guidelines?
- **Retrieval Evaluation**: Are the retrieved sources relevant to the query?

You can learn more about how evaluation works in LlamaIndex in our [module guides](/module_guides/evaluating/root.md).

## Related concepts

You may be interested in [analyzing the cost of your application](/understanding/evaluating/cost_analysis/root.md) if you are making calls to a hosted, remote LLM.

```{toctree}
---
maxdepth: 1
hidden: true
---
/understanding/evaluating/cost_analysis/root.md
```
