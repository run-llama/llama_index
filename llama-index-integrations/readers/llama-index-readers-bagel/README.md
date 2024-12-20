# LlamaIndex Readers Integration: Bagel

```bash
pip install llama-index-readers-bagel
```

## Bagel Loader

## Usage

```python
from llama_index.core.schema import Document
from llama_index.readers.bagel import BagelReader

# Initialize BagelReader with the collection name
reader = BagelReader(collection_name="example_collection")

# Load data from Bagel
documents = reader.load_data(
    query_vector=None,
    query_texts=["example text"],
    limit=10,
    where=None,
    where_document=None,
    include=["documents", "embeddings"],
)
```

## Features

- Retrieve documents, embeddings, and metadata efficiently.
- Filter results based on specified conditions.
- Specify what data to include in the retrieved results.

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
