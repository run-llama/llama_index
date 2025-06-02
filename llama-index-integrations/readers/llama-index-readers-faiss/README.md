# LlamaIndex Readers Integration: Faiss

## Overview

Faiss Reader retrieves documents through an existing in-memory Faiss index. These documents can then be used in a downstream LlamaIndex data structure. If you wish to use Faiss itself as an index to organize documents, insert documents, and perform queries on them, please use VectorStoreIndex with FaissVectorStore.

### Installation

You can install Faiss Reader via pip:

```bash
pip install llama-index-readers-faiss
```

## Usage

```python
from llama_index.readers.faiss import FaissReader

# Initialize FaissReader with an existing Faiss Index object
reader = FaissReader(index="<Faiss Index Object>")

# Load data from Faiss
documents = reader.load_data(
    query="<Query Vector>",  # 2D numpy array of query vectors
    id_to_text_map={"<ID>": "<Text>"},  # A map from IDs to text
    k=4,  # Number of nearest neighbors to retrieve
    separate_documents=True,  # Whether to return separate documents
)
```

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index).
