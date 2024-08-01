# LlamaIndex Vector_Stores Integration: Alibaba Cloud OpenSearch

Please refer to the [notebook](../../../docs/docs/examples/vector_stores/AlibabaCloudOpenSearchIndexDemo.ipynb) for usage of AlibabaCloud OpenSearch as vector store in LlamaIndex.

## Example Usage

```sh
pip install llama-index
pip install llama-index-vector-stores-alibabacloud-opensearch
```

```python
# Connect to existing instance
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.alibabacloud_opensearch import (
    AlibabaCloudOpenSearchStore,
    AlibabaCloudOpenSearchConfig,
)

config = AlibabaCloudOpenSearchConfig(
    endpoint="***",
    instance_id="***",
    username="your_username",
    password="your_password",
    table_name="llama",
)

vector_store = AlibabaCloudOpenSearchStore(config)

# Create index from existing stored vectors
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine()
response = query_engine.query(
    "What did the author study prior to working on AI?"
)
print(response)
```
