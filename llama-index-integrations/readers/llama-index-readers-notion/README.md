# LlamaIndex Readers Integration: Notion

## Overview

Notion Page Reader enables loading data from Notion pages. It constructs queries to retrieve pages based on page IDs or from a specified Notion database.

### Installation

You can install Notion Reader via pip:

```bash
pip install llama-index-readers-notion
```

### Usage

```python
from llama_index.readers.notion import NotionPageReader

# Initialize NotionPageReader
reader = NotionPageReader(integration_token="<Integration Token>")

# Load data from Notion
documents = reader.load_data(
    page_ids=["<Page ID 1>", "<Page ID 2>"],  # List of page IDs to load
    database_id="<Database ID>",  # Database ID from which to load page IDs
)
```

Implementation for Notion reader can be found [here](https://docs.llamaindex.ai/en/stable/examples/data_connectors/NotionDemo/)

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
