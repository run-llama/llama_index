# LlamaCloud Index + Retriever

LlamaCloud is a new generation of managed parsing, ingestion, and retrieval services, designed to bring production-grade context-augmentation to your LLM and RAG applications.

Currently, LlamaCloud supports

- Managed Ingestion API, handling parsing and document management
- Managed Retrieval API, configuring optimal retrieval for your RAG system

## Access

We are opening up a private beta to a limited set of enterprise partners for the managed ingestion and retrieval API. If you’re interested in centralizing your data pipelines and spending more time working on your actual RAG use cases, come [talk to us.](https://www.llamaindex.ai/contact)

If you have access to LlamaCloud, you can visit [LlamaCloud](https://cloud.llamaindex.ai) to sign in and get an API key.

## Setup

First, make sure you have the latest LlamaIndex version installed.

```
pip uninstall llama-index  # run this if upgrading from v0.9.x or older
pip install -U llama-index --upgrade --no-cache-dir --force-reinstall
```

The `llama-index-indices-managed-llama-cloud` package is included with the above install, but you can also install directly

```
pip install -U llama-index-indices-managed-llama-cloud
```

## Usage

You can create an index on LlamaCloud using the following code:

```python
import os

os.environ[
    "LLAMA_CLOUD_API_KEY"
] = "llx-..."  # can provide API-key in env or in the constructor later on

from llama_index.core import SimpleDirectoryReader
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex

# create a new index
index = LlamaCloudIndex.from_documents(
    documents,
    "my_first_index",
    project_name="default",
    api_key="llx-...",
    verbose=True,
)

# connect to an existing index
index = LlamaCloudIndex("my_first_index", project_name="default")
```

You can also configure a retriever for managed retrieval:

```python
# from the existing index
index.as_retriever()

# from scratch
from llama_index.indices.managed.llama_cloud import LlamaCloudRetriever

retriever = LlamaCloudRetriever("my_first_index", project_name="default")
```

And of course, you can use other index shortcuts to get use out of your new managed index:

```python
query_engine = index.as_query_engine(llm=llm)

chat_engine = index.as_chat_engine(llm=llm)
```

## Retriever Settings

A full list of retriever settings/kwargs is below:

- `dense_similarity_top_k`: Optional[int] -- If greater than 0, retrieve `k` nodes using dense retrieval
- `sparse_similarity_top_k`: Optional[int] -- If greater than 0, retrieve `k` nodes using sparse retrieval
- `enable_reranking`: Optional[bool] -- Whether to enable reranking or not. Sacrifices some speed for accuracy
- `rerank_top_n`: Optional[int] -- The number of nodes to return after reranking initial retrieval results
- `alpha` Optional[float] -- The weighting between dense and sparse retrieval. 1 = Full dense retrieval, 0 = Full sparse retrieval.
