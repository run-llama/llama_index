Storage
============

LlamaIndex provides a high-level interface for ingesting, indexing, and querying your external data.
By default, LlamaIndex hides away the complexities and let you query your data in under 5 lines of code.
Under the hood, LlamaIndex also supports a swappable **storage layer** that allows you to customize:

- Document Stores: where ingested documents (i.e., `Node` objects) are stored,
- Index Stores: where index metadata are stored,
- Vector Stores: where embedding vectors are stored.

.. toctree::
   :maxdepth: 1
   :caption: Storage

   storage/save_load.md
   storage/customization.md
   storage/docstores.md
   storage/index_stores.md
   storage/vector_stores.md