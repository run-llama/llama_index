# LlamaIndex Vector_Stores Integration: Moorcheh

Welcome to the Moorcheh Vector Store that integrates Llama-Index.

This module introduces support for [**Moorcheh**](https://github.com/mjfekri/moorcheh-python-sdk), a semantic vector database developed by **EdgeAI Innovations**. Moorcheh enables fast and intelligent document retrieval using hybrid scoring and generative answering capabilities.
The integration is implemented in accordance with the standard vector store interface defined by **LlamaIndex** and supports all core methods including `add`, `query`, `delete`, and `generate_answer`.

To see the integration in action, refer to the demonstration notebook: [Google Colab Demo](https://colab.research.google.com/drive/1iUoMpNYcJxmu1xTySMNJZBPbOQIkUEEs?usp=sharing).

## Getting started

To begin using the Moorcheh vector store, make sure to install the necessary packages:

```
pip install llama_index
pip install moorcheh_sdk
```

## Example Usage

Here is a simple example demonstrating how to use the Moorcheh integration with LlamaIndex:

```
from llama_index.core import VectorStoreIndex
from llama_index.llama_index_integrations.vector_stores.llama_index_vector_stores_moorcheh.llama_index.vector_stores-moorcheh import base, init, utils

api_key = os.environ["MOORCHEH_API_KEY"]

documents = SimpleDirectoryReader("./your-directory").load_data()
__all__ = ["MoorchehVectorStore"]

# Creates a Moorcheh Vector Store with the following parameters
# For text-based namespaces, set namespace_type to "text" and vector_dimension to None
# For vector-based namespaces, set namespace_type to "vector" and vector_dimension to the dimension of your uploaded vectors
vector_store = MoorchehVectorStore(api_key=api_key, namespace="llamaindex_moorcheh", namespace_type="text", vector_dimension=None, add_sparse_vector=False, batch_size=100)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

query_engine = index.as_query_engine()
response = query_engine.query("Which company has had the highest revenue in 2025 and why?")

display(Markdown(f"<b>{response}</b>"))
print("\n\n================================\n\n", response, "\n\n================================\n\n")
```
