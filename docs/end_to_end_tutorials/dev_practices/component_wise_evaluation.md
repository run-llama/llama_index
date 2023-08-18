# Component Wise Evaluation
## Evaluating Retrieval

### BEIR dataset

BEIR is useful for benchmarking if a particular retrieval model generalize well to niche domains in a zero-shot setting (without fine-tuning).

Here is a notebook showing how the BEIR dataset can be used with your retrieval pipeline.

We will be adding more methods to evaluate retrieval soon. This includes evaluating retrieval on your own dataset.

## Evaluating the Query Engine

### Standard Metrics

Against annotated datasets, whether your own data or an academic benchmark, there are a number of standard metrics that it helps to be aware of:

1. **Exact Match (EM):** The percentage of queries that are answered exactly correctly.
2. **F1:** The percentage of queries that are answered exactly correctly or with a small edit distance (e.g. 1-2 words).
3. **Recall:** The percentage of queries that are answered correctly, regardless of the number of answers returned.
4. **Precision:** The percentage of queries that are answered correctly, divided by the number of answers returned.

This [towardsdatascience article](https://towardsdatascience.com/ranking-evaluation-metrics-for-recommender-systems-263d0a66ef54) covers more technical metrics like NDCG, MAP and MRR in greater depth.

### HotpotQA Dataset

The HotpotQA dataset is useful for evaluating queries that require multiple retrieval steps.

Limitations:

1. HotpotQA is evaluated on a Wikipedia corpus. LLMs, especially GPT4, tend to have memorized information from Wikipedia relatively well. Hence, the benchmark is not particularly good for evaluating retrieval + rerank systems with knowledgeable models like GPT4.

### Examples

```{toctree}
---
maxdepth: 1
---
/examples/evaluation/BeirEvaluation.ipynb
/examples/evaluation/HotpotQADistractor.ipynb
```