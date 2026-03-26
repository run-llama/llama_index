# Apache Solr Vector Store for LlamaIndex

A LlamaIndex VectorStore using Apache Solr as the backend.

## Install

```bash
pip install llama-index
pip install llama-index-vector-stores-solr
```

## Solr References

- [Getting Started](https://solr.apache.org/guide/solr/latest/getting-started/introduction.html)
- [Deployment Guide](https://solr.apache.org/guide/solr/latest/deployment-guide/solr-control-script-reference.html)
- [Indexing Guide](https://solr.apache.org/guide/solr/latest/indexing-guide/schema-elements.html)
- [Query Guide](https://solr.apache.org/guide/solr/latest/query-guide/common-query-parameters.html)
- [KNN/Vector Search Support](https://solr.apache.org/guide/solr/latest/getting-started/tutorial-vectors.html)
- [Docker Setup](https://solr.apache.org/guide/solr/latest/deployment-guide/solr-in-docker.html)

## Quickstart

### Imports

```python
from llama_index.vector_stores.solr import (
    ApacheSolrVectorStore,
    SyncSolrClient,
    AsyncSolrClient,
)
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    MockEmbedding,
)
import pytest
from llama_index.core import Settings
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.core.schema import NodeWithScore
```

### Setup Vector Store

```python
SOLR_COLLECTION_URL = "http://localhost:8983/solr/my_collection"  # assumes a solr collection is running here

sync_client = SyncSolrClient(base_url=SOLR_COLLECTION_URL)
async_client = AsyncSolrClient(base_url=SOLR_COLLECTION_URL)

vector_store = ApacheSolrVectorStore(
    sync_client=sync_client,
    async_client=async_client,
    nodeid_field="id",
    content_field="content_txt_en",  # store content in a text field searchable by BM25
    embedding_field="vector_field",  # dense vector field configured in Solr schema
    metadata_to_solr_field_mapping=[
        ("author", "author_s"),
    ],
    text_search_fields=["content_txt_en"],
)
```

### Create Index and Query

```python
# Dummy Documents
docs = [
    Document(
        text="Apache Solr integrates with LlamaIndex.",
        metadata={"author": "alice"},
    ),
    Document(
        text="Vector search lets you find semantically similar text.",
        metadata={"author": "bob"},
    ),
]

# Create the index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# NOTE: This will use the default embedding model set at Settings.embed_model
# Configure your own if you wish to do so.
# Alternatively this Vectorstore can be used with the IngestionPipeline as well.
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
```

### BM25 Search

- This is a naive implementation ideally create a retriever with [BaseRetriever](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/base/base_retriever.py#L34)

```python
def simple_bm25(vector_store, query_str="semantic search"):
    results = vector_store.query(
        VectorStoreQuery(
            mode=VectorStoreQueryMode.TEXT_SEARCH,
            query_str=query_str,
            sparse_top_k=2,
        )
    )
    return [
        NodeWithScore(node=node, score=score)
        for node, score in zip(
            results.nodes, results.similarities or [], strict=True
        )
    ]


bm25_search_results = simple_bm25(vector_store, query_str="semantic search")
```

### Dense Vector Search

- This is a naive implementation ideally create a retriever with [BaseRetriever](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/base/base_retriever.py#L34)

```python
# Dense Vector Search
def simple_vector_search(vector_store, query_str="semantic search"):
    retriever = index.as_retriever(similarity_top_k=1)
    return retriever.retrieve("semantic search")


vector_search_results = simple_vector_search(vector_store, "semantic search")
```

### Query Engine

```python
query_engine = index.as_query_engine(similarity_top_k=2)
# NOTE: Will use default embedding and LLM model set at Settings
# Configure your own if you wish to do so.
res = query_engine.query("semantic search")
```
