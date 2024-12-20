# LlamaIndex Readers Integration: Weaviate

## Overview

The Weaviate Reader retrieves documents from Weaviate through vector lookup. It allows you to specify a class name and properties to retrieve from documents, or to provide a custom GraphQL query. You can choose to receive separate Document objects per document or concatenate retrieved documents into one Document.

### Installation

You can install the Weaviate Reader via pip:

```bash
pip install llama-index-readers-weaviate
```

### Usage

```python
from llama_index.readers.weaviate import WeaviateReader

# Initialize WeaviateReader with host and optional authentication
reader = WeaviateReader(
    host="<Weaviate Host>", auth_client_secret="<Authentication Client Secret>"
)

# Load data from Weaviate
documents = reader.load_data(
    class_name="<Class Name>", properties=["property 1", "property 2"]
)
```

You can follow this tutorial to learn more on how to use [Weaviate Reader](https://docs.llamaindex.ai/en/stable/examples/data_connectors/WeaviateDemo/)

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
