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

## Using a remote backend

By default, LlamaIndex uses a local filesystem to load and save files. However, you can override this by passing a `fsspec.AbstractFileSystem` object.

Here's a simple example, instantiating a vector store:
```python
import dotenv
import s3fs
import os
dotenv.load_dotenv("../../../.env")

# load documents
documents = SimpleDirectoryReader('../../../examples/paul_graham_essay/data/').load_data()
print(len(documents))
index = GPTVectorStoreIndex.from_documents(documents)
```

At this point, everything has been the same. Now - let's instantiate a S3 filesystem and save / load from there.

```python
# set up s3fs
AWS_KEY = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET = os.environ['AWS_SECRET_ACCESS_KEY']
R2_ACCOUNT_ID = os.environ['R2_ACCOUNT_ID']

assert AWS_KEY is not None and AWS_KEY != ""

s3 = s3fs.S3FileSystem(
   key=AWS_KEY,
   secret=AWS_SECRET,
   endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
   s3_additional_kwargs={'ACL': 'public-read'}
)

# save index to remote blob storage
index.set_index_id("vector_index")
# this is {bucket_name}/{index_name}
index.storage_context.persist('llama-index/storage_demo', fs=s3)

# load index from s3
sc = StorageContext.from_defaults(persist_dir='llama-index/storage_demo', fs=s3)
index2 = load_index_from_storage(sc, 'vector_index')
```

By default, if you do not pass a filesystem, we will assume a local filesystem.