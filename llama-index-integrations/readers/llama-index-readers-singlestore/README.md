# SingleStore Loader

```bash
pip install llama-index-readers-singlestore
```

The SingleStore Loader retrieves a set of documents from a specified table in a SingleStore database. The user initializes the loader with database information and then provides a search embedding for retrieving similar documents.

## Usage

Here's an example usage of the SingleStoreReader:

```python
from llama_index.readers.singlestore import SingleStoreReader

# Initialize the reader with your SingleStore database credentials and other relevant details
reader = SingleStoreReader(
    scheme="mysql",
    host="localhost",
    port="3306",
    user="username",
    password="password",
    dbname="database_name",
    table_name="table_name",
    content_field="text",
    vector_field="embedding",
)

# The search_embedding is an embedding representation of your query_vector.
# Example search_embedding:
#   search_embedding=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
search_embedding = [n1, n2, n3, ...]

# load_data fetches documents from your SingleStore database that are similar to the search_embedding.
# The top_k argument specifies the number of similar documents to fetch.
documents = reader.load_data(search_embedding=search_embedding, top_k=5)
```
