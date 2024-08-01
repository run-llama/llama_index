# LlamaIndex Readers Integration: Qdrant

## Overview

The Qdrant Reader allows you to retrieve documents from existing Qdrant collections. Qdrant is a similarity search engine that helps you efficiently search and retrieve similar items from large datasets based on vector embeddings.

For more detailed information about Qdrant, visit [Qdrant](qdrant.io)

### Installation

You can install the Qdrant Reader via pip:

```bash
pip install llama-index-readers-qdrant
```

### Usage

```python
from llama_index.readers.qdrant import QdrantReader

# Initialize QdrantReader
reader = QdrantReader(
    location="<Qdrant Location>",
    url="<Qdrant URL>",
    port="<Port>",
    grpc_port="<gRPC Port>",
    prefer_grpc="<Prefer gRPC>",
    https="<Use HTTPS>",
    api_key="<API Key>",
    prefix="<URL Prefix>",
    timeout="<Timeout>",
    host="<Host>",
)

# Load data from Qdrant
documents = reader.load_data(
    collection_name="<Collection Name>",
    query_vector=[0.1, 0.2, 0.3],
    should_search_mapping={"text_field": "text"},
    must_search_mapping={"text_field": "text"},
    must_not_search_mapping={"text_field": "text"},
    rang_search_mapping={"text_field": {"gte": 0.1, "lte": 0.2}},
    limit=10,
)
```

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
