# Storage

## Concept

LlamaIndex provides a high-level interface for ingesting, indexing, and querying your external data.

Under the hood, LlamaIndex also supports swappable **storage components** that allows you to customize:

- **Document stores**: where ingested documents (i.e., `Node` objects) are stored,
- **Index stores**: where index metadata are stored,
- **Vector stores**: where embedding vectors are stored.

The Document/Index stores rely on a common Key-Value store abstraction, which is also detailed below.

LlamaIndex supports persisting data to any storage backend supported by [fsspec](https://filesystem-spec.readthedocs.io/en/latest/index.html). 
We have confirmed support for the following storage backends:

- Local filesystem
- AWS S3
- Cloudflare R2


![](/_static/storage/storage.png)

## Usage Pattern

To use these storage abstractions, you need to define a `StorageContext` object:

```python
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore
from llama_index.storage import StorageContext

# create storage context using default stores
storage_context = StorageContext.from_defaults(
    docstore=SimpleDocumentStore(),
    vector_store=SimpleVectorStore(),
    index_store=SimpleIndexStore(),
)
```

More details on customization/persistence can be found in the guides below.


```{toctree}
---
maxdepth: 1
---
customization.md
save_load.md
```



## Modules

We offer in-depth guides on the different storage components.

```{toctree}
---
maxdepth: 1
---
vector_stores.md
docstores.md
index_stores.md
kv_stores.md
```
