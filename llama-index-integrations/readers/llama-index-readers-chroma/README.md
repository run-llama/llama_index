# LlamaIndex Readers Integration: Chroma

## Overview

Chroma Reader is a tool designed to retrieve documents from existing persisted Chroma collections. Chroma is a framework for managing document collections and their associated embeddings efficiently.

### Installation

You can install Chroma Reader via pip:

```bash
pip install llama-index-readers-chroma
```

## Usage

```python
from llama_index.core.schema import Document
from llama_index.readers.chroma import ChromaReader

# Initialize ChromaReader with the collection name and optional parameters
reader = ChromaReader(
    collection_name="<Your Collection Name>",
    persist_directory="<Directory Path>",  # Optional: Directory where the collection is persisted
    chroma_api_impl="rest",  # Optional: Chroma API implementation (default: "rest")
    chroma_db_impl=None,  # Optional: Chroma DB implementation (default: None)
    host="localhost",  # Optional: Host for Chroma DB (default: "localhost")
    port=8000,  # Optional: Port for Chroma DB (default: 8000)
)

# Load data from Chroma collection
documents = reader.load_data(
    query_embedding=None,  # Provide query embedding if searching by embeddings
    limit=10,  # Number of results to retrieve
    where=None,  # Filter condition for metadata
    where_document=None,  # Filter condition for document
    query=["search term"],  # Provide query text if searching by text
)
```

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
