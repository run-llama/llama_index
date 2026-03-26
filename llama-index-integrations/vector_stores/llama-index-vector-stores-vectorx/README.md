# LlamaIndex Vector_Stores Integration: VectorX

## 🚀 Installation

```bash
pip install llama-index-core vecx
```

## Usage

```Python
from llama_index.vector_stores.vectorx import VectorXVectorStore
from llama_index.core import VectorStoreIndex

# Initialize store
store = VectorXVectorStore.from_params(
    api_token="YOUR_API_TOKEN",
    encryption_key="YOUR_ENCRYPTION_KEY",
    index_name="my-index",
    dimension=768,  # must match your embeddings
)

# Wrap with LlamaIndex
index = VectorStoreIndex.from_vector_store(store)

# Insert/query as usual
query_engine = index.as_query_engine()
response = query_engine.query("What is VectorX?")
print(response)
```
