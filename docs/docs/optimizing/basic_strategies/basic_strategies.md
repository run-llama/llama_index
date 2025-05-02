# Basic Strategies

There are many easy things to try, when you need to quickly squeeze out extra performance and optimize your RAG workflow.

## Prompt Engineering

If you're encountering failures related to the LLM, like hallucinations or poorly formatted outputs, then this
should be one of the first things you try.

Some tasks are listed below, from simple to advanced.

1. Try inspecting the prompts used in your RAG workflow (e.g. the question–answering prompt) and customizing it.

- [Customizing Prompts](../../examples/prompts/prompt_mixin.ipynb)
- [Advanced Prompts](../../examples/prompts/advanced_prompts.ipynb)

2. Try adding **prompt functions**, allowing you to dynamically inject few-shot examples or process the injected inputs.

- [Advanced Prompts](../../examples/prompts/advanced_prompts.ipynb)

## Embeddings

Choosing the right embedding model plays a large role in overall performance.

- Maybe you need something better than the default `text-embedding-ada-002` model from OpenAI?
- Maybe you want to scale to a local server?
- Maybe you need an embedding model that works well for a specific language?

Beyond OpenAI, many options existing for embedding APIs, running your own embedding model locally, or even hosting your own server.

A great resource to check on the current best overall embeddings models is the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard), which ranks embeddings models on over 50 datasets and tasks.

**NOTE:** Unlike an LLM (which you can change at any time), if you change your embedding model, you must re-index your data. Furthermore, you should ensure the same embedding model is used for both indexing and querying.

We have a list of [all supported embedding model integrations](../../module_guides/models/embeddings.md).

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

1. Use a vector database that has a hybrid search functionality (see [our complete list of supported vector stores](../../module_guides/storing/vector_stores.md)).
2. Set up a local hybrid search mechanism with BM25.

Relevant guides with both approaches can be found below:

- [BM25 Retriever](../../examples/retrievers/bm25_retriever.ipynb)
- [Reciprocal Rerank Query Fusion](../../examples/retrievers/reciprocal_rerank_fusion.ipynb)
- [Weaviate Hybrid Search](../../examples/vector_stores/WeaviateIndexDemo-Hybrid.ipynb)
- [Pinecone Hybrid Search](../../examples/vector_stores/PineconeIndexDemo-Hybrid.ipynb)
- [Milvus Hybrid Search](../../examples/vector_stores/MilvusHybridIndexDemo.ipynb)

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

If you are using an advanced LLM like GPT-4, and your [vector database supports filtering](../../module_guides/storing/vector_stores.md), you can get the LLM to write filters automatically at query time, using an `AutoVectorRetriever`.

- [Vector Store Guide](../../module_guides/indexing/vector_store_guide.ipynb)

## Document/Node Usage

Take a look at our in-depth guides for more details on how to use Documents/Nodes.

- [Documents Usage](../../module_guides/loading/documents_and_nodes/usage_documents.md)
- [Nodes Usage](../../module_guides/loading/documents_and_nodes/usage_nodes.md)
- [Metadata Extraction](../../module_guides/loading/documents_and_nodes/usage_metadata_extractor.md)

## Multi-Tenancy RAG

Multi-Tenancy in RAG systems is crucial for ensuring data security. It enables users to access exclusively their indexed documents, thereby preventing unauthorized sharing and safeguarding data privacy. Search operations are confined to the user's own data, protecting sensitive information. Implementation can be achieved with `VectorStoreIndex` and `VectorDB` providers through Metadata Filters.

Refer the guides below for more details.

- [Multi Tenancy RAG](../../examples/multi_tenancy/multi_tenancy_rag.ipynb)

For detailed guidance on implementing Multi-Tenancy RAG with LlamaIndex and Qdrant, refer to the [blog post](https://qdrant.tech/documentation/tutorials/llama-index-multitenancy/) released by Qdrant.
