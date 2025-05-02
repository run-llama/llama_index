# Astra DB Loader

```bash
pip install llama-index-readers-astra-db
```

The Astra DB Loader returns a set of documents retrieved from Astra DB.
The user initializes the loader with an Astra DB index. They then pass in a vector.

## Usage

Here's an example usage of the AstraDBReader.

```python
from openai import OpenAI


# Get the credentials for Astra DB
api_endpoint = "https://324<...>f1c.astra.datastax.com"
token = "AstraCS:<...>"

# EXAMPLE: OpenAI embeddings
client = OpenAI(api_key="sk-<...>")

# Call OpenAI (or generate embeddings another way)
response = client.embeddings.create(
    input="Your text string goes here", model="text-embedding-ada-002"
)

# Get the embedding
query_vector = response.data[0].embedding

# Initialize the Reader object
from llama_index.readers.astra_db import AstraDBReader

# Your Astra DB Account will provide you with the endpoint URL and Token
reader = AstraDBReader(
    collection_name="astra_v_table",
    token=token,
    api_endpoint=api_endpoint,
    embedding_dimension=len(query_vector),
)

# Fetch data from the reader
documents = reader.load_data(vector=query_vector, limit=5)
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).

> **Note**: Please see the AstraDB documentation [here](https://docs.datastax.com/en/astra/astra-db-vector/clients/python.html).
