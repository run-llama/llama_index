---
title: Rerankers
---

## Concept

A reranker takes the top-`k` nodes returned by an initial retriever (usually a dense vector search, keyword search, or a hybrid of the two) and re-orders them based on a stronger, and typically slower, measure of query-document relevance. Reranking is one of the highest-leverage knobs in a RAG pipeline: initial retrievers are tuned for recall over a large corpus, and a reranker lets you recover precision before nodes reach the LLM.

In LlamaIndex, rerankers are implemented as node postprocessors that run between retrieval and response synthesis.

## Usage pattern

```python
from llama_index.core.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L6-v2", top_n=3
)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker],
)
response = query_engine.query("What did the author do in college?")
```

The retriever fetches a wider set (`similarity_top_k=10`) and the reranker prunes it down to the best `top_n=3` before the LLM sees them.

## Choosing a reranker

- No API key, local, strong quality: [`SentenceTransformerRerank`](/python/framework/module_guides/querying/node_postprocessors/node_postprocessors#sentencetransformerrerank) is the recommended default. Pair with a modern cross-encoder like [`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) (stronger, multilingual) or [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) (smaller, faster). Browse more models on the [Sentence Transformers cross-encoder list](https://sbert.net/docs/cross_encoder/pretrained_models.html) and the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
- Hosted API, minimal setup: `CohereRerank`, `JinaRerank`, `VoyageAIRerank`, `MixedbreadAIRerank`. Good options when you'd rather not run inference yourself.
- Highest quality, latency is not critical: `LLMRerank`, `RankGPTRerank`, `RankLLMRerank`. These use an LLM itself to judge relevance.
- Multimodal late interaction (documents as images): [`ColPaliRerank`](/python/examples/node_postprocessor/colpalirerank).
- Text late interaction: [`ColbertRerank`](/python/examples/node_postprocessor/colbertrerank). Useful when the query and document contain narrow technical terms.

For the full reference including every integration and every model, see the [Node Postprocessors guide](/python/framework/module_guides/querying/node_postprocessors/node_postprocessors). Rerankers are one category of node postprocessors.

## Further reading

- [Node Postprocessors guide](/python/framework/module_guides/querying/node_postprocessors/node_postprocessors): every supported reranker with code examples.
- [Sentence Transformers cross-encoder documentation](https://sbert.net/docs/cross_encoder/pretrained_models.html): speed/accuracy tradeoffs for open-source models.
- [Advanced retrieval](/python/framework/optimizing/advanced_retrieval/advanced_retrieval): how reranking fits into a broader retrieval strategy.
