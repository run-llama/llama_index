# LlamaIndex Readers Integration: txtai

## Overview

The txtai Reader retrieves documents through an existing in-memory txtai index. These documents can then be used in downstream LlamaIndex data structures. If you wish to use txtai itself as an index to organize documents, insert documents, and perform queries on them, please use VectorStoreIndex with TxtaiVectorStore.

### Installation

You can install the txtai Reader via pip:

```bash
pip install llama-index-readers-txtai
```

### Usage

```python
from llama_index.readers.txtai import TxtaiReader

# Initialize TxtaiReader with an existing txtai index
reader = TxtaiReader(index="<txtai Index object>")

# Load data from txtai index
documents = reader.load_data(
    query="<Query Vector>",
    id_to_text_map={"<ID>": "<Text>"},
    k=4,
    separate_documents=True,
)
```

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index).
