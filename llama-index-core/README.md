# LlamaIndex Core

The core python package to the LlamaIndex library. Core classes and abstractions
represent the foundational building blocks for LLM applications, most notably,
RAG. Such building blocks include abstractions for LLMs, Vector Stores, Embeddings,
Storage, Callables and several others.

We've designed the core library so that it can be easily extended through subclasses.
Building LLM applications with LlamaIndex thus involves building with LlamaIndex
core as well as with the LlamaIndex [integrations](https://github.com/run-llamae/llama_index/tree/main/llama-index-integrations) needed for your application.
