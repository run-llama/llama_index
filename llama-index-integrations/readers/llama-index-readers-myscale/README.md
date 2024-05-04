# LlamaIndex Readers Integration: Myscale

## Overview

MyScale Reader allows loading data from a MyScale backend. It constructs a query to retrieve documents based on a given query vector and additional search parameters.

### Installation

You can install Myscale Reader via pip:

```bash
pip install llama-index-readers-myscale
```

### Usage

```python
from llama_index.readers.myscale import MyScaleReader

# Initialize MyScaleReader
reader = MyScaleReader(
    myscale_host="<MyScale Host>",  # MyScale host address
    username="<Username>",  # Username to login
    password="<Password>",  # Password to login
    database="<Database Name>",  # Database name (default: 'default')
    table="<Table Name>",  # Table name (default: 'llama_index')
    index_type="<Index Type>",  # Index type (default: "IVFLAT")
    metric="<Metric>",  # Metric to compute distance (default: 'cosine')
    batch_size=32,  # Batch size for inserting documents (default: 32)
    index_params=None,  # Index parameters for MyScale (default: None)
    search_params=None,  # Search parameters for MyScale query (default: None)
)

# Load data from MyScale
documents = reader.load_data(
    query_vector=[0.1, 0.2, 0.3],  # Query vector
    where_str="<Where Condition>",  # Where condition string (default: None)
    limit=10,  # Number of results to return (default: 10)
)
```

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
