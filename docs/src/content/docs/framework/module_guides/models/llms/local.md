---
title: Using local models
---

LlamaIndex can run entirely against local models. This is useful when you don't want to send data to a hosted API, when you need to work offline, or when you want predictable cost. The typical fully-local stack is:

- LLM: llama.cpp, vLLM, Hugging Face Transformers, or Ollama.
- Embeddings: [`HuggingFaceEmbedding`](/python/framework/module_guides/models/embeddings#local-embedding-models), which runs any [Sentence Transformers](https://sbert.net/) model locally (with optional ONNX / OpenVINO acceleration on CPU).
- Reranker (optional): [`SentenceTransformerRerank`](/python/framework/module_guides/models/rerankers) with a cross-encoder like `cross-encoder/ms-marco-MiniLM-L6-v2` or `Qwen/Qwen3-Reranker-0.6B`.
- Vector store: the default `SimpleVectorStore` (persisted to disk), or any self-hosted integration (Chroma, Qdrant, Postgres/pgvector, Milvus, etc.).

## Quickest path

Follow the [local starter tutorial](/python/framework/getting_started/starter_example_local), which walks through installing Ollama, `llama-index-llms-ollama`, and `llama-index-embeddings-huggingface`, then builds a full local RAG pipeline.

For a persistent local stack backed by Chroma, see the [fully-local RAG cookbook](/python/examples/cookbooks/local_rag_with_chroma_and_ollama).

## Available local LLM integrations

LlamaIndex supports local LLMs via several integration packages:

- `llama-index-llms-llama-cpp`: direct llama.cpp bindings.
- `llama-index-llms-huggingface`: Hugging Face Transformers (any causal LM from the Hub).
- `llama-index-llms-vllm`: vLLM for high-throughput self-hosted serving.
- `llama-index-llms-openai-like`: any server that exposes an OpenAI-compatible API (text-generation-webui, LM Studio, vLLM, etc.).
- `llama-index-llms-ollama`: Ollama (llama3, mistral, qwen, etc.).

Browse the full list on [LlamaHub](https://llamahub.ai).
