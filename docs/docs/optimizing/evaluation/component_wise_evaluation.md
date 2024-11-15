# Component Wise Evaluation

To do more in-depth evaluation of your workflow, it helps to break it down into an evaluation of individual components.

For instance, a particular failure case may be due to a combination of not retrieving the right documents and also the LLM misunderstanding the context and hallucinating an incorrect result. Being able to isolate and deal with these issues separately can help reduce complexity and guide you in a step-by-step manner to a more satisfactory overall result.

## Utilizing public benchmarks

When doing initial model selection, it helps to look at how well the model is performing on a standardized, diverse set of domains or tasks.

A useful benchmark for embeddings is the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

## Evaluating Retrieval

### BEIR dataset

BEIR is useful for benchmarking if a particular retrieval model generalize well to niche domains in a zero-shot setting.

Since most publicly-available embedding and retrieval models are already benchmarked against BEIR (e.g. through the MTEB benchmark), utilizing BEIR is more helpful when you have a unique model that you want to evaluate.

For instance, after fine-tuning an embedding model on your dataset, it may be helpful to view whether and by how much its performance degrades on a diverse set of domains. This can be an indication of how much data drift may affect your retrieval accuracy, such as if you add documents to your RAG system outside of your fine-tuning training distribution.

Here is a notebook showing how the BEIR dataset can be used with your retrieval flow.

- [BEIR Evaluation](../../examples/evaluation/BeirEvaluation.ipynb)

We will be adding more methods to evaluate retrieval soon. This includes evaluating retrieval on your own dataset.

## Evaluating the Query Engine Components (e.g. Without Retrieval)

In this case, we may want to evaluate how specific components of a query engine (one which may generate sub-questions or follow-up questions) may perform on a standard benchmark. It can help give an indication of how far behind or ahead your retrieval flow is compared to alternate flows or models.

### HotpotQA Dataset

The HotpotQA dataset is useful for evaluating queries that require multiple retrieval steps.

Example:

- [HotpotQA Eval](../../examples/evaluation/HotpotQADistractor.ipynb)

Limitations:

1. HotpotQA is evaluated on a Wikipedia corpus. LLMs, especially GPT4, tend to have memorized information from Wikipedia relatively well. Hence, the benchmark is not particularly good for evaluating retrieval + rerank systems with knowledgeable models like GPT4.
