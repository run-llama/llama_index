# Building RAG from Scratch (Lower-Level)

This doc is a hub for showing how you can build RAG and agent-based apps using only lower-level abstractions (e.g. LLMs, prompts, embedding models), and without using more "packaged" out of the box abstractions.

Out of the box abstractions include:
- High-level ingestion code e.g. `VectorStoreIndex.from_documents`
- High-level query and retriever code e.g. `VectorStoreIndex.as_retriever()` and `VectorStoreIndex.as_query_engine()`
- High-level agent abstractions e.g. `OpenAIAgent`

Instead of using these, the goal here is to educate users on what's going on under the hood. By showing you the underlying algorithms for constructing RAG and agent pipelines, you can then be empowered to create your own custom LLM workflows (while still using LlamaIndex abstractions at any level of granularity that makes sense). 

We show how to build an app from scratch, component by component. For the sake of focus, each tutorial will show how to build a specific component from scratch while using out-of-the-box abstractions for other components. **NOTE**: This is a WIP document, we're in the process of fleshing this out! 

## Building Ingestion from Scratch
This tutorial shows how you can define an ingestion pipeline into a vector store.
```{toctree}
---
maxdepth: 1
---
/examples/low_level/ingestion.ipynb
```

## Building Vector Retrieval from Scratch
This tutorial shows you how to build a retriever to query a vector store.

```{toctree}
---
maxdepth: 1
---
/examples/low_level/retrieval.ipynb
```


## Building Response Synthesis from Scratch
This tutorial shows you how to use the LLM to synthesize results given a set of retrieved context. Deals with context overflows, async calls, and source citations!
```{toctree}
---
maxdepth: 1
---
/examples/low_level/retrieval.ipynb
```


