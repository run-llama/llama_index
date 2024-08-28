# LlamaIndex Vector_Stores Integration: MariaDB

With the release of MariaDB 11.6 Vector Preview, the MariaDB relational database introduced the long-awaited vector search functionality.
Thus now it can be used as a fully-functional vector store in LlamaIndex.
Please note, however, that the latest MariaDB version is only an Alpha release, which means that it may crash unexpectedly.

To learn more about the feature, check the [Vector Overview](https://mariadb.com/kb/en/vector-overview/) in the MariaDB docs.

## Installation

```shell
pip install llama-index-vector-stores-mariadb
```

## Usage

```python
from llama_index.vector_stores.mariadb import MariaDBVectorStore

vector_store = MariaDBVectorStore.from_params(
    host="localhost",
    port=3306,
    user="llamaindex",
    password="password",
    database="vectordb",
    table_name="llama_index_vectorstore",
    embed_dim=1536,  # OpenAI embedding dimension
)
```

## Development

### Running Integration Tests

A suite of integration tests is available to verify the MariaDB vector store integration.
The test suite needs a MariaDB database with vector search support up and running, if not found the tests are skipped.
To facilitate that, a sample `docker-compose.yaml` file is provided, so you can simply do:

```shell
docker compose -f tests/docker-compose.yaml up

pytest -v

# Clean up when you finish testing
docker compose -f tests/docker-compose.yaml down
```
