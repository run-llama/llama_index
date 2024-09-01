# CouchDB Loader

```bash
pip install llama-index-readers-couchdb
```

This loader loads documents from CouchDB. The loader currently supports CouchDB 3.x
using the CouchDB3 python wrapper from https://github.com/n-vlahovic/couchdb3
The user specifies a CouchDB instance to initialize the reader. They then specify
the database name and query params to fetch the relevant docs.

## Usage

Here's an example usage of the SimpleCouchDBReader.

```python
import os

from llama_index.readers.couchdb import SimpleCouchDBReader

host = "<host>"
port = "<port>"
db_name = "<db_name>"
# query is passed into db.find()
query_str = "{ couchdb_find_sytax_json }"
reader = SimpleCouchDBReader(host, port)
documents = reader.load_data(db_name, query=query_str)
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
