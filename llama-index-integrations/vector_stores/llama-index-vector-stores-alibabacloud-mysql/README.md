# LlamaIndex Vector_Stores Integration: Alibaba Cloud MySQL

Alibaba Cloud MySQL supports vector search functionality. This package provides a vector store implementation that allows you to use Alibaba Cloud MySQL as a vector database in LlamaIndex.

## Installation

```shell
pip install llama-index-vector-stores-alibabacloud-mysql
```

## Usage

```python
from llama_index.vector_stores.alibabacloud_mysql import (
    AlibabaCloudMySQLVectorStore,
)

vector_store = AlibabaCloudMySQLVectorStore(
    host="your-instance-endpoint.mysql.rds.aliyuncs.com",
    port=3306,
    user="llamaindex",
    password="password",
    database="vectordb",
)
```

Or using the from_params class method:

```python
from llama_index.vector_stores.alibabacloud_mysql import (
    AlibabaCloudMySQLVectorStore,
)

vector_store = AlibabaCloudMySQLVectorStore.from_params(
    host="your-instance-endpoint.mysql.rds.aliyuncs.com",
    port=3306,
    user="llamaindex",
    password="password",
    database="vectordb",
)
```

## Features

- Full compatibility with Alibaba Cloud MySQL 8.0+
- Support for vector indexing and fast similarity search
- Filtering support with metadata queries
- Easy integration with LlamaIndex
- Uses mysql.connector for direct database connections
- Enhanced validation for vector function support

## Development

### Running Integration Tests

A suite of integration tests is available to verify the Alibaba Cloud MySQL vector store integration.
The test suite needs an Alibaba Cloud MySQL database with vector search support up and running. If not found, the tests are skipped.

```shell
pytest -v
```
