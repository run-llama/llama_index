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

## Building Ingestion/Retrieval from Scratch (Open-Source/Local Components)

This tutoral shows you how to build an ingestion/retrieval pipeline using only
open-source components.

```{toctree}
---
maxdepth: 1
---
/examples/low_level/oss_ingestion_retrieval.ipynb
```

## Building a (Very Simple) Vector Store from Scratch

If you want to learn more about how vector stores work, here's a tutorial showing you how to build a very simple vector store capable of dense search + metadata filtering.

Obviously not a replacement for production databases.

```{toctree}
---
maxdepth: 1
---
/examples/low_level/vector_store.ipynb
```

## Building Response Synthesis from Scratch

This tutorial shows you how to use the LLM to synthesize results given a set of retrieved context. Deals with context overflows, async calls, and source citations!

```{toctree}
---
maxdepth: 1
---
/examples/low_level/response_synthesis.ipynb
```

## Building Evaluation from Scratch

Learn how to build common LLM-based eval modules (correctness, faithfulness) using LLMs and prompt modules; this will help you define your own custom evals!

```{toctree}
---
maxdepth: 1
---
/examples/low_level/evaluation.ipynb
```

## Building Advanced RAG from Scratch

These tutorials will show you how to build advanced functionality beyond the basic RAG pipeline. Especially helpful for advanced users with custom workflows / production needs.

### Building Hybrid Search from Scratch

Hybrid search is an advanced retrieval feature supported by many vector databases. It allows you to combine **dense** retrieval with **sparse** retrieval with matching keywords.

```{toctree}
---
maxdepth: 1
---
Building Hybrid Search from Scratch </examples/vector_stores/qdrant_hybrid.ipynb>
```

### Building a Router from Scratch

Beyond the standard RAG pipeline, this takes you one step towards automated decision making with LLMs by showing you how to build a router module from scratch.

```{toctree}
---
maxdepth: 1
---
/examples/low_level/router.ipynb
```

### Building RAG Fusion Retriever from Scratch

Here we show you how to build an advanced retriever capable of query-rewriting, ensembling, dynamic retrieval.

```{toctree}
---
maxdepth: 1
---
/examples/low_level/fusion_retriever.ipynb
```
