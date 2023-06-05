# Index Stores

Index stores contains lightweight index metadata (i.e. additional state information created when building an index).

See the [API Reference](/reference/storage/index_store.rst) for more details.

### Simple Index Store
By default, LlamaIndex uses a simple index store backed by an in-memory key-value store.
They can be persisted to (and loaded from) disk by calling `index_store.persist()` (and `SimpleIndexStore.from_persist_path(...)` respectively).


### MongoDB Index Store
Similarly to document stores, we can also use `MongoDB` as the storage backend of the index store.


```python
from llama_index.storage.index_store import MongoIndexStore


# create (or load) index store
index_store = MongoIndexStore.from_uri(uri="<mongodb+srv://...>")

# create storage context
storage_context = StorageContext.from_defaults(index_store=index_store)

# build index
index = VectorStoreIndex(nodes, storage_context=storage_context)

# or alternatively, load index
index = load_index_from_storage(storage_context)
```

Under the hood, `MongoIndexStore` connects to a fixed MongoDB database and initializes new collections (or loads existing collections) for your index metadata.
> Note: You can configure the `db_name` and `namespace` when instantiating `MongoIndexStore`, otherwise they default to `db_name="db_docstore"` and `namespace="docstore"`.

Note that it's not necessary to call `storage_context.persist()` (or `index_store.persist()`) when using an `MongoIndexStore`
since data is persisted by default. 

You can easily reconnect to your MongoDB collection and reload the index by re-initializing a `MongoIndexStore` with an existing `db_name` and `collection_name`.


