# LlamaIndex Vector_Stores Integration: Hologres

Please refer to the [notebook](../../../docs/docs/examples/vector_stores/HologresDemo.ipynb) for usage of Hologres as vector store in LlamaIndex.

## Example Usage

```sh
pip install llama-index
pip install llama-index-vector-stores-hologres
```

```python
# Connect to existing instance
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.hologres import HologresVectorStore

vector_store = HologresVectorStore.from_param(
    host="***",
    port=80,
    user="***",
    password="***",
    database="***",
    table_name="***",
    embedding_dimension=1536,
    pre_delete_table=True,
)

# Create index from existing stored vectors
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine()
response = query_engine.query(
    "What did the author study prior to working on AI?"
)
print(response)
```
