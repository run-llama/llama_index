# LlamaCloudIndex + LlamaCloudRetriever

LlamaCloud is a new generation of managed parsing, ingestion, and retrieval services, designed to bring production-grade context-augmentation to your LLM and RAG applications.

Currently, LlamaCloud supports

- Managed Ingestion API, handling parsing and document management
- Managed Retrieval API, configuring optimal retrieval for your RAG system

For additional documentation on LlamaCloud and this integration in particular, please reference our [official LlamaCloud docs](https://docs.cloud.llamaindex.ai/llamacloud/guides/framework_integration).

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


## Composite Retrieval Usage

Once you've setup multiple indexes that are ingesting various forms of data, you may want to create an application that can query over the data across all of your indices.

This is where you can use the `LlamaCloudCompositeRetriever` class. The following snippet shows you how to setup the composite retriever:

```python
import os
from llama_cloud import CompositeRetrievalMode, RetrieverPipeline
from llama_index.indices.managed.llama_cloud import (
    LlamaCloudIndex,
    LlamaCloudCompositeRetriever,
)

llama_cloud_api_key = os.environ["LLAMA_CLOUD_API_KEY"]
project_name = "Essays"

# Setup your indices
pg_documents = SimpleDirectoryReader("./examples/data/paul_graham").load_data()
pg_index = LlamaCloudIndex.from_documents(
    documents=pg_documents,
    name="PG Index",
    project_name=project_name,
    api_key=llama_cloud_api_key,
)

sama_documents = SimpleDirectoryReader(
    "./examples/data/sam_altman"
).load_data()
sama_index = LlamaCloudIndex.from_documents(
    documents=sama_documents,
    name="Sam Index",
    project_name=project_name,
    api_key=llama_cloud_api_key,
)

retriever = LlamaCloudCompositeRetriever(
    name="Essays Retriever",
    project_name=project_name,
    api_key=llama_cloud_api_key,
    # If a Retriever named "Essays Retriever" doesn't already exist, one will be created
    create_if_not_exists=True,
    # CompositeRetrievalMode.FULL will query each index individually and globally rerank results at the end
    mode=CompositeRetrievalMode.FULL,
    rerank_top_n=5,
)

# Add the above indices to the composite retriever
# Carefully craft the description as this is used internally to route a query to an attached sub-index when CompositeRetrievalMode.ROUTING is used
retriever.add_index(pg_index, description="A collection of Paul Graham essays")
retriever.add_index(
    sama_index, description="A collection of Sam Altman essays"
)

# Start retrieving context for your queries
# async .aretrieve() is also available
nodes = retriever.retrieve("What does YC do?")
```

### Composite Retrieval related parameters
There are a few parameters that are specific to tuning the composite retrieval parameters:
- `mode`: `Optional[CompositeRetrievalMode]` -- Can either be `CompositeRetrievalMode.FULL` or `CompositeRetrievalMode.ROUTING`
    - `full`: In this mode, all attached sub-indices will be queried and reranking will be executed across all nodes received from these sub-indices.
    - `routing`: In this mode, an agent determines which sub-indices are most relevant to the provided query (based on the sub-index's `name` & `description` you've provided) and only queries those indices that are deemed relevant. Only the nodes from that chosen subset of indices are then reranked before being returned in the retrieval response.
- `rerank_top_n`: `Optional[int]` -- Determines how many nodes to return after re-ranking across the nodes retrieved from all indices
