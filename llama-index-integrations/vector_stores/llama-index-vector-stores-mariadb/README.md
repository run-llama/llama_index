# LlamaIndex Vector_Stores Integration: MariaDB

Starting with version `11.7.1`, the MariaDB relational database has vector search functionality integrated.
Thus now it can be used as a fully-functional vector store in LlamaIndex.

To learn more about the feature in MariaDB, check its [Vector Overview documentation](https://mariadb.com/kb/en/vector-overview/).

Please note that versions before `0.3.0` of this package are not compatible with MariaDB 11.7 and later.
They are compatible only with the one-off `MariaDB 11.6 Vector` preview release which used a slightly different syntax.

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
The test suite needs a MariaDB database with vector search support up and running. If not found, the tests are skipped.
To facilitate that, a sample `docker-compose.yaml` file is provided, so you can simply do:

```shell
docker compose -f tests/docker-compose.yaml up

pytest -v

# Clean up when you finish testing
docker compose -f tests/docker-compose.yaml down
```
