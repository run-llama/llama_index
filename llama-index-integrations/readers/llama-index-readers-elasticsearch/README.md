# LlamaIndex Readers Integration: Elasticsearch

## Overview

Elasticsearch (or Opensearch) Reader over REST API is a tool designed to read documents from an Elasticsearch or Opensearch index using the basic search API. These documents can then be utilized in downstream LlamaIndex data structures.

### Installation

You can install Elasticsearch (or Opensearch) Reader via pip:

```bash
pip install llama-index-readers-elasticsearch
```

## Usage

```python
from llama_index.core.schema import Document
from llama_index.readers.elasticsearch import ElasticsearchReader

# Initialize ElasticsearchReader
reader = ElasticsearchReader(
    endpoint="<Your Elasticsearch/Opensearch Endpoint>",
    index="<Index Name>",
    httpx_client_args={
        "timeout": 10
    },  # Optional additional arguments for the httpx.Client
)

# Load data from Elasticsearch
documents = reader.load_data(
    field="<Field Name>",  # Field in the document to retrieve text from
    query={"query": {"match_all": {}}},  # Elasticsearch JSON query DSL object
    embedding_field="<Embedding Field>",  # Field for embeddings (optional)
)
```

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
