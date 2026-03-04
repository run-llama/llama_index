# LlamaIndex Vector_Stores Integration: Volcengine MySQL

This integration provides a `VolcengineMySQLVectorStore` that leverages the native vector index capabilities of Volcano engine Cloud Database for MySQL. It allows you to use RDS for MySQL as a fully-functional vector store in LlamaIndex for high-performance similarity search.

To learn more about the native vector feature in Volcengine RDS for MySQL, you can refer to its official documentation. This integration is specifically designed for Volcengine's enhanced MySQL and is not intended for standard community MySQL instances.

## Installation

```shell
pip install llama-index-vector-stores-volcengine-mysql
```

## Usage

```python
from llama_index.vector_stores.volcengine_mysql import (
    VolcengineMySQLVectorStore,
)

# Initialize the vector store
vector_store = VolcengineMySQLVectorStore.from_params(
    host="your-rds-instance-host",
    port=3306,
    user="your-username",
    password="your-password",
    database="your-database",
    table_name="llama_index_vector_store",
    embed_dim=1536,  # Example: OpenAI embedding dimension
)
```

### Requirements

To use this integration, you need a Volcengine RDS for MySQL instance with the following features enabled:

- The `loose_vector_index_enabled` parameter set to `ON`.
- Support for `VECTOR(N)` data type for columns.
- Support for `VECTOR INDEX` (HNSW) for indexing.
- Access to vector functions like `TO_VECTOR(...)`, `L2_DISTANCE(...)`, and `COSINE_DISTANCE(...)`.
