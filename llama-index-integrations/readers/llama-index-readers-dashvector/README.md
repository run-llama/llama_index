# LlamaIndex Readers Integration: Dashvector

## Overview

DashVector Reader is a tool designed to retrieve documents from DashVector clusters efficiently.

### Installation

You can install DashVector Reader via pip:

```bash
pip install llama-index-readers-dashvector
```

To use DashVector, you must have an API key. Here are the [installation instructions](https://help.aliyun.com/document_detail/2510223.html)

## Usage

```python
from llama_index.core.schema import Document
from llama_index.readers.dashvector import DashVectorReader

# Initialize DashVectorReader with the API key and cluster endpoint
reader = DashVectorReader(
    api_key="<Your API Key>", endpoint="<Cluster Endpoint>"
)

# Load data from DashVector
documents = reader.load_data(
    collection_name="<Collection Name>",
    vector=[0.1, 0.2, 0.3],  # Query vector
    topk=10,  # Number of results to return
    separate_documents=True,  # Whether to return separate documents
    filter=None,  # Optional: Filter conditions
    include_vector=True,  # Whether to include the embedding in the response
    output_fields=None,  # Optional: Fields Filter
)
```

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
