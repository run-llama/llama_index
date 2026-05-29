# LlamaIndex Readers Integration: CockroachDB

Load rows from an existing [CockroachDB](https://www.cockroachlabs.com/) table or arbitrary SELECT into LlamaIndex `Document` objects.

## Installation

```bash
pip install llama-index-readers-cockroachdb
```

## Usage

```python
from llama_index.readers.cockroachdb import CockroachDBReader

reader = CockroachDBReader.from_params(
    host="localhost",
    port=26257,
    database="defaultdb",
    user="root",
    password="",
    sslmode="disable",
)

# Mode 1: table + columns
docs = reader.load_data(
    table="articles",
    text_column="body",
    metadata_columns=["id", "author", "tag"],
    id_column="id",
)

# Mode 2: parameterized query
docs = reader.load_data(
    query="SELECT id, body, author FROM articles WHERE author = :a",
    text_column="body",
    metadata_columns=["id", "author"],
    id_column="id",
    params={"a": "alice"},
)
```

The reader uses the `cockroachdb+psycopg2` SQLAlchemy dialect for serialization-retry support. Pair it with [`llama-index-vector-stores-cockroachdb`](../../vector_stores/llama-index-vector-stores-cockroachdb) if you want to index the rows you load.
