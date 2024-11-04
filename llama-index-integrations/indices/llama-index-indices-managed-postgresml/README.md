# LlamaIndex Managed Integration: PostgresML

PostgresML provides an all in one platform for production ready RAG applications.

# Setup

First, make sure you have the latest LlamaIndex version installed and a connection string to your PostgresML database.

If you don't already have a connection string, you can get one on [postgresml.org](https://postgresml.org).

```
pip install llama-index-indices-managed-postgresml
```

# Usage

Getting started is easy!

```python
import os

os.environ[
    "PGML_DATABASE_URL"
] = "..."  # Can provide in the environment or constructor later on

from llama_index.core import Document
from llama_index.indices.managed.postgresml import PostgresMLIndex

# Create an index
index = PostgresMLIndex.from_documents(
    "llama-index-test-1", [Document.example()]
)

# Connect to an index
index = PostgresMLIndex("llama-index-test-1")
```

You can use the index as a retriever

```python
# Create a retriever from an index
retriever = index.as_retriever()

results = retriever.retrieve("What managed index is the best?")
print(results)
```

You can also use the index as a query engine

```python
# Create an engine from an index
query_engine = index.as_query_engine()

response = retriever.retrieve("What managed index is the best?")
print(response)
```
