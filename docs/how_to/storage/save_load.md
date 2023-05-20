# Persisting & Loading Data

## Persisting Data
By default, LlamaIndex stores data in-memory, and this data can be explicitly persisted if desired:
```python
storage_context.persist(persist_dir="<persist_dir>")
```
This will persist data to disk, under the specified `persist_dir` (or `./storage` by default).

Multiple indexes can be persisted and loaded from the same directory, assuming you keep track of index ID's for loading.

User can also configure alternative storage backends (e.g. `MongoDB`) that persist data by default.
In this case, calling `storage_context.persist()` will do nothing.

## Loading Data
To load data, user simply needs to re-create the storage context using the same configuration (e.g. pass in the same `persist_dir` or vector store client).

```python
storage_context = StorageContext.from_defaults(
    docstore=SimpleDocumentStore.from_persist_dir(persist_dir="<persist_dir>"),
    vector_store=SimpleVectorStore.from_persist_dir(persist_dir="<persist_dir>"),
    index_store=SimpleIndexStore.from_persist_dir(persist_dir="<persist_dir>"),
)
```

We can then load specific indices from the `StorageContext` through some convenience functions below.


```python
from llama_index import load_index_from_storage, load_indices_from_storage, load_graph_from_storage

# load a single index
# need to specify index_id if multiple indexes are persisted to the same directory
index = load_index_from_storage(storage_context, index_id="<index_id>") 

# don't need to specify index_id if there's only one index in storage context
index = load_index_from_storage(storage_context) 

# load multiple indices
indices = load_indices_from_storage(storage_context) # loads all indices
indices = load_indices_from_storage(storage_context, index_ids=[index_id1, ...]) # loads specific indices

# load composable graph
graph = load_graph_from_storage(storage_context, root_id="<root_id>") # loads graph with the specified root_id
```

Here's the full [API Reference on saving and loading](/reference/storage/indices_save_load.rst).



