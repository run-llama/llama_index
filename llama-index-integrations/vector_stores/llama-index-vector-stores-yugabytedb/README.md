# YugabyteDB Vector Store

A LlamaIndex vector store using YugabyteDB as the backend.

## Usage

Pre-requisite:

```bash
pip install llama-index-vector-stores-yugabytedb
```

A minimal example:

```python
from llama_index.vector_stores.yugabytedb import YBVectorStore

vector_store = YBVectorStore.from_params(
    host="localhost",
    user="yugabyte",
    password="yugabyte",
    port=5433,
    load_balance="True",
    database="yugabyte",
    table_name="test_table",
    schema_name="test_schema",
    embed_dim=1536,
)
```

> **Note**: Please see the YugabyteDB psycopge2 driver documentation for more yugabytedb specific parameters [here](https://docs.yugabyte.com/preview/drivers-orms/python/yugabyte-psycopg2/#step-2-set-up-the-database-connection).
