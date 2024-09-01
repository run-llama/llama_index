# LlamaIndex Readers Integration: ArangoDB

```bash
pip install llama-index-readers-arango-db
```

This loader loads documents from [ArangoDB](https://github.com/arangodb/arangodb?tab=readme-ov-file#arangodb). The user specifies an ArangoDB instance to
initialize the reader. They then specify the collection name and query parameters to
fetch the relevant docs.

## Usage

Here's an example usage of the SimpleArangoDBReader.

```python
import os

from llama_index.readers.arango_db import SimpleArangoDBReader

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

A demo notebook is available [here](https://colab.research.google.com/github/arangodb/interactive_tutorials/blob/master/notebooks/example_output/Langchain_Full_output.ipynb).

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
