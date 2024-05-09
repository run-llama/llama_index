# LlamaIndex Readers Integration: Metal

## Overview

Metal Reader is designed to load data from the Metal Vector store, which provides search functionality based on query embeddings and filters. It retrieves documents from the Metal index associated with the provided API key, client ID, and index ID.

### Installation

You can install Metal Reader via pip:

```bash
pip install llama-index-readers-metal
```

To use Metal Reader, you must have a vector store first. Follow this to create a metal vector store, [Setup Metal Vector Store](https://docs.llamaindex.ai/en/stable/examples/vector_stores/MetalIndexDemo/)

### Usage

```python
from llama_index.readers.metal import MetalReader

# Initialize MetalReader
reader = MetalReader(
    api_key="<Metal API Key>",
    client_id="<Metal Client ID>",
    index_id="<Metal Index ID>",
)

# Load data from Metal
documents = reader.load_data(
    limit=10,  # Number of results to return
    query_embedding=[0.1, 0.2, 0.3],  # Query embedding for search
    filters={"field": "value"},  # Filters to apply to the search
    separate_documents=True,  # Whether to return separate documents
)
```

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
