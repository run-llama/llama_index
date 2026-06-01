---
title: Basic Strategies
---

There are many easy things to try, when you need to quickly squeeze out extra performance and optimize your RAG workflow.

## Prompt Engineering

If you're encountering failures related to the LLM, like hallucinations or poorly formatted outputs, then this
should be one of the first things you try.

Some tasks are listed below, from simple to advanced.

1. Try inspecting the prompts used in your RAG workflow (e.g. the question–answering prompt) and customizing it.

- [Customizing Prompts](/python/examples/prompts/prompt_mixin)
- [Advanced Prompts](/python/examples/prompts/advanced_prompts)

2. Try adding **prompt functions**, allowing you to dynamically inject few-shot examples or process the injected inputs.

- [Advanced Prompts](/python/examples/prompts/advanced_prompts)

## Embeddings

Choosing the right embedding model plays a large role in overall performance.

- Maybe you want a stronger or more recent model than OpenAI's `text-embedding-ada-002` (still the `OpenAIEmbedding` default for backwards compatibility)?
- Maybe you want to run a model locally rather than call a hosted API?
- Maybe you need an embedding model that works well for a specific language?

Beyond OpenAI, many options exist for embedding APIs, running your own embedding model locally, or hosting your own server.

The [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) is the best place to compare current models across tons of datasets and tasks. At the time of writing, popular open-source picks include:

- Fast / small baseline (~22-33M params): [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) or [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5). Tiny, fast, still solid for many workloads.
- Mid-size, recent architecture (~150M params): [`lightonai/DenseOn`](https://huggingface.co/lightonai/DenseOn), released April 2026. Sits between the small baselines and the Qwen3 family in size, worth trying if you want recent state-of-the-art retrieval without going to a 0.6B+ model.
- Strong all-around, multilingual (~0.6-4B params): [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (or `-4B` for higher accuracy). Competitive on MTEB, small enough to serve on a single GPU.
- Frictionless hosted API: `OpenAIEmbedding(model="text-embedding-3-small")`.

All of the local/open-source options above can be used via [`HuggingFaceEmbedding`](/python/framework/module_guides/models/embeddings#local-embedding-models) and benefit from ONNX / OpenVINO acceleration on CPU; see the [embeddings guide](/python/framework/module_guides/models/embeddings) for details.

**NOTE:** Unlike an LLM (which you can change at any time), if you change your embedding model, you must re-index your data. Furthermore, you should ensure the same embedding model is used for both indexing and querying.

We have a list of [all supported embedding model integrations](/python/framework/module_guides/models/embeddings).

## Chunk Sizes

Depending on the type of data you are indexing, or the results from your retrieval, you may want to customize the chunk size or chunk overlap.

When documents are ingested into an index, they are split into chunks with a certain amount of overlap. The default chunk size is 1024, while the default chunk overlap is 20.

Changing either of these parameters will change the embeddings that are calculated. A smaller chunk size means the embeddings are more precise, while a larger chunk size means that the embeddings may be more general, but can miss fine-grained details.

We have done our own [initial evaluation on chunk sizes here](https://blog.llamaindex.ai/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5).

Furthermore, when changing the chunk size for a vector index, you may also want to increase the `similarity_top_k` parameter to better represent the amount of data to retrieve for each query.

Here is a full example:

```
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings

documents = SimpleDirectoryReader("./data").load_data()

Settings.chunk_size = 512
Settings.chunk_overlap = 50

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine(similarity_top_k=4)
```

Since we halved the default chunk size, the example also doubles the `similarity_top_k` from the default of 2 to 4.

## Hybrid Search

Hybrid search is a common term for retrieval that involves combining results from both semantic search (i.e. embedding similarity) and keyword search.

Embeddings are not perfect, and may fail to return text chunks with matching keywords in the retrieval step.

The solution to this issue is often hybrid search. In LlamaIndex, there are two main ways to achieve this:

1. Use a vector database that has a hybrid search functionality (see [our complete list of supported vector stores](/python/framework/module_guides/storing/vector_stores)).
2. Set up a local hybrid search mechanism with BM25.

Relevant guides with both approaches can be found below:

- [BM25 Retriever](/python/examples/retrievers/bm25_retriever)
- [Reciprocal Rerank Query Fusion](/python/examples/retrievers/reciprocal_rerank_fusion)
- [Weaviate Hybrid Search](/python/examples/vector_stores/weaviateindexdemo-hybrid)
- [Pinecone Hybrid Search](/python/examples/vector_stores/pineconeindexdemo-hybrid)
- [Milvus Hybrid Search](/python/examples/vector_stores/milvushybridindexdemo)

## Reranking

Reranking is one of the highest-leverage knobs for RAG quality. The retriever returns a wider set of candidates (`similarity_top_k=10` or more) and a reranker (a stronger but slower model) re-orders them so the best nodes reach the LLM. It's often the difference between "retrieval looks roughly right" and "retrieval is actually answering the question."

The frictionless recipe uses a local cross-encoder via [`SentenceTransformerRerank`](/python/framework/module_guides/querying/node_postprocessors/node_postprocessors#sentencetransformerrerank); no API key, runs anywhere:

```python
from llama_index.core.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L6-v2", top_n=3
)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker],
)
```

For stronger quality on multilingual content, swap the model for `Qwen/Qwen3-Reranker-0.6B`. For a hosted-API option with minimal setup, use `CohereRerank`, `JinaRerank`, or `VoyageAIRerank`. For the highest quality when latency isn't critical, an LLM-based reranker (`LLMRerank`, `RankGPTRerank`) can outperform cross-encoders at the cost of extra LLM calls.

See the [node postprocessors guide](/python/framework/module_guides/querying/node_postprocessors/node_postprocessors) for the full list and the [rerankers overview](/python/framework/module_guides/models/rerankers) for a decision tree.

## Metadata Filters

Before throwing your documents into a vector index, it can be useful to attach metadata to them. While this metadata can be used later on to help track the sources to answers from the `response` object, it can also be used at query time to filter data before performing the top-k similarity search.

Metadata filters can be set manually, so that only nodes with the matching metadata are returned:

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

documents = [
    Document(text="text", metadata={"author": "LlamaIndex"}),
    Document(text="text", metadata={"author": "John Doe"}),
]

filters = MetadataFilters(
    filters=[ExactMatchFilter(key="author", value="John Doe")]
)

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(filters=filters)
```

If you are using an advanced LLM like GPT-4, and your [vector database supports filtering](/python/framework/module_guides/storing/vector_stores), you can get the LLM to write filters automatically at query time, using an `AutoVectorRetriever`.

- [Vector Store Guide](/python/framework/module_guides/indexing/vector_store_guide)

## Document/Node Usage

Take a look at our in-depth guides for more details on how to use Documents/Nodes.

- [Documents Usage](/python/framework/module_guides/loading/documents_and_nodes/usage_documents)
- [Nodes Usage](/python/framework/module_guides/loading/documents_and_nodes/usage_nodes)
- [Metadata Extraction](/python/framework/module_guides/loading/documents_and_nodes/usage_metadata_extractor)

## Multi-Tenancy RAG

Multi-Tenancy in RAG systems is crucial for ensuring data security. It enables users to access exclusively their indexed documents, thereby preventing unauthorized sharing and safeguarding data privacy. Search operations are confined to the user's own data, protecting sensitive information. Implementation can be achieved with `VectorStoreIndex` and `VectorDB` providers through Metadata Filters.

Refer the guides below for more details.

- [Multi Tenancy RAG](/python/examples/multi_tenancy/multi_tenancy_rag)

For detailed guidance on implementing Multi-Tenancy RAG with LlamaIndex and Qdrant, refer to the [blog post](https://qdrant.tech/documentation/tutorials/llama-index-multitenancy/) released by Qdrant.
