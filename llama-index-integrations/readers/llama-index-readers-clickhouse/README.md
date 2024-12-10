# LlamaIndex Readers Integration: ClickHouse

## Overview

ClickHouse Reader is a tool designed to retrieve documents from ClickHouse databases efficiently.

## Installation

You can install ClickHouse Reader via pip:

```bash
pip install llama-index-readers-clickhouse
```

## Usage

```python
from llama_index.core.schema import Document
from llama_index.readers.clickhouse import ClickHouseReader

# Initialize ClickHouseReader with the connection details and configuration
reader = ClickHouseReader(
    clickhouse_host="<ClickHouse Host>",
    username="<Username>",
    password="<Password>",
    clickhouse_port=8123,  # Optional: Default port is 8123
    database="<Database Name>",
    engine="MergeTree",  # Optional: Default engine is "MergeTree"
    table="<Table Name>",
    index_type="NONE",  # Optional: Default index type is "NONE"
    metric="cosine",  # Optional: Default metric is "cosine"
    batch_size=1000,  # Optional: Default batch size is 1000
    index_params=None,  # Optional: Index parameters
    search_params=None,  # Optional: Search parameters
)

# Load data from ClickHouse
documents = reader.load_data(
    query_vector=[0.1, 0.2, 0.3],  # Query vector
    where_str=None,  # Optional: Where condition string
    limit=10,  # Optional: Number of results to return
)
```

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
