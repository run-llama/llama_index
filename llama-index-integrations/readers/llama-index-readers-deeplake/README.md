# LlamaIndex Readers Integration: Deeplake

## Overview

DeepLake Reader is a tool designed to retrieve documents from existing DeepLake datasets efficiently.

### Installation

You can install DeepLake Reader via pip:

```bash
pip install llama-index-readers-deeplake
```

To use Deeplake Reader, you must have an API key. Here are the [installation instructions](https://docs.activeloop.ai/storage-and-credentials/user-authentication)

## Usage

```python
from llama_index.core.schema import Document
from llama_index.readers.deeplake import DeepLakeReader

# Initialize DeepLakeReader with the token
reader = DeepLakeReader(token="<Your DeepLake Token>")

# Load data from DeepLake
documents = reader.load_data(
    query_vector=[0.1, 0.2, 0.3],  # Query vector
    dataset_path="<Path to Dataset>",  # Path to the DeepLake dataset
    limit=4,  # Number of results to return
    distance_metric="l2",  # Distance metric
)
```

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
