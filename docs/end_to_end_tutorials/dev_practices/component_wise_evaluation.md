# Component Wise Evaluation
## Evaluating Retrieval

### BEIR dataset

BEIR is useful for benchmarking if a particular retrieval model generalize well to niche domains in a zero-shot setting (without fine-tuning).

We will be adding more methods to evaluate retrieval soon. This includes evaluating retrieval on your own dataset.

## Evaluating the Query Engine

### Standard Metrics

Against annotated datasets, whether your own data or an academic benchmark, there are a number of standard metrics that it helps to be aware of:

1. **Exact Match (EM):** The percentage of queries that are answered exactly correctly.
2. **F1:** The percentage of queries that are answered exactly correctly or with a small edit distance (e.g. 1-2 words).
3. **Recall:** The percentage of queries that are answered correctly, regardless of the number of answers returned.
4. **Precision:** The percentage of queries that are answered correctly, divided by the number of answers returned.

There are more technical metrics such as MRR, MAP, etc. that are useful for academic benchmarks, but we will not cover them here.

### HotpotQA Dataset

The HotpotQA dataset is useful for evaluating queries that require multiple retrieval steps.

Limitations:

1. HotpotQA is evaluated on a Wikipedia corpus. LLMs, especially GPT4, tend to have memorized information from Wikipedia relatively well. Hence, the benchmark is not particularly good for evaluating retrieval + rerank systems with knowledgeable models like GPT4.