# LlamaIndex Readers Integration: Arango Db

This loader loads documents from ArangoDB. The user specifies a ArangoDB instance to
initialize the reader. They then specify the collection name and query params to
fetch the relevant docs.

## Usage

Here's an example usage of the SimpleArangoDBReader.

```python
from llama_index.core.readers import download_loader
import os

SimpleArangoDBReader = download_loader("SimpleArangoDBReader")

host = "<host>"
db_name = "<db_name>"
collection_name = "<collection_name>"
# query_dict is passed into db.collection.find()
query_dict = {}
# Attribute of interests to load, by default ["text"]
field_names = ["title", "description"]
reader = SimpleArangoDBReader(host)  # or pass ArangoClient
documents = reader.load_data(
    username,
    password,
    db_name,
    collection_name,
    query_dict=query_dict,
    field_names=field_names,
)
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent. See [here](https://github.com/run-llama/llama-hub/tree/main/llama_hub) for examples.
