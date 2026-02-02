# LlamaIndex Readers Integration: Zvec

## Overview

Zvec Reader is a tool designed to retrieve documents from Zvec collection efficiently.

### Installation

You can install Zvec Reader via pip:

```bash
pip install llama-index-readers-zvec
```

## Usage

```python
from llama_index.core.schema import Document
from llama_index.readers.zvec import ZvecReader

# Initialize ZvecReader with the collection path
reader = ZvecReader(path="<your collection path>")

# Load data from Zvec
documents = reader.load_data(
    vector=[0.1, 0.2, 0.3],  # Query vector
    topk=10,  # Number of results to return
    filter=None,  # Optional: Filter conditions
    include_vector=True,  # Whether to include the embedding in the response
    output_fields=None,  # Optional: Fields Filter
)
```

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index).
