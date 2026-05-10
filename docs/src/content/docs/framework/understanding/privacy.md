---
title: Privacy and Security
---

By default, LlamaIndex tutorials use hosted APIs from OpenAI for both LLM generation and embedding, which means your documents and queries leave your machine. For many projects that's fine: OpenAI, Anthropic, Cohere, and similar providers publish data-handling policies and offer enterprise agreements with stronger data-protection terms. For projects where data cannot leave your infrastructure, LlamaIndex can run the full RAG pipeline locally.

## Running LlamaIndex fully locally

A fully-local stack looks like:

- LLM: a local runtime such as llama.cpp, vLLM, Hugging Face Transformers, or Ollama.
- Embeddings: [`HuggingFaceEmbedding`](/python/framework/module_guides/models/embeddings#local-embedding-models), which runs any [Sentence Transformers](https://sbert.net/) model on your own hardware.
- Reranker (optional but recommended): [`SentenceTransformerRerank`](/python/framework/module_guides/models/rerankers), a cross-encoder that runs locally, no API key required.
- Vector store: the default in-memory `SimpleVectorStore` (optionally persisted to disk via `StorageContext.persist()`) or a self-hosted store like Chroma, Qdrant, Postgres/pgvector, or Milvus.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SentenceTransformerRerank

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.llm = Ollama(model="llama3.1", request_timeout=360.0)

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L6-v2", top_n=3
)
query_engine = index.as_query_engine(
    similarity_top_k=10, node_postprocessors=[reranker]
)
```

No API key, no outbound network calls from the embedding, reranking, or retrieval steps.

### Where to go next

- [Local starter tutorial](/python/framework/getting_started/starter_example_local): step-by-step walkthrough with Ollama + Hugging Face embeddings.
- [Fully-local RAG cookbook (Chroma + Ollama)](/python/examples/cookbooks/local_rag_with_chroma_and_ollama): a persistent local pipeline with evaluation.
- [Using local LLMs](/python/framework/module_guides/models/llms/local): list of local LLM integrations.
- [Embeddings: local models](/python/framework/module_guides/models/embeddings#local-embedding-models), including ONNX / OpenVINO acceleration for CPU-only environments.

## Data privacy with hosted APIs

If you use a hosted LLM or embedding provider, data privacy is governed by that provider's terms. OpenAI, Anthropic, Cohere, Voyage, and similar each publish their own policies and offer enterprise agreements. LlamaIndex itself does not store your data, but it will send requests to whichever providers you configure.

Common mitigations short of going fully local:

- Use Azure OpenAI, AWS Bedrock, Google Vertex AI, or a similar cloud where you already have a data-protection agreement in place.
- Self-host the embedding and reranker steps (they're small) while keeping only the LLM call hosted. This is often the biggest risk-reduction per unit of effort.
- Scrub PII from documents before indexing via the [`PIINodePostprocessor`](/python/framework/module_guides/querying/node_postprocessors/node_postprocessors#beta-piinodepostprocessor).

## Vector store privacy

LlamaIndex integrates with many vector stores, each with its own data-handling policy. Self-hosted options (`SimpleVectorStore`, Chroma, Qdrant, Postgres/pgvector, Milvus, Weaviate when run locally) keep embeddings on your infrastructure. Managed options (Pinecone, hosted Weaviate, Zilliz, etc.) store embeddings on the provider's infrastructure under that provider's terms. Consult the [vector stores guide](/python/framework/module_guides/storing/vector_stores) for the full list.
