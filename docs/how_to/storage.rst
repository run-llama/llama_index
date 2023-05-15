💾 Storage
============

LlamaIndex provides a high-level interface for ingesting, indexing, and querying your external data.
By default, LlamaIndex hides away the complexities and let you query your data in `under 5 lines of code </how_to/storage/customization.md>`_.


Under the hood, LlamaIndex also supports swappable **storage components** that allows you to customize:

- **Document stores**: where ingested documents (i.e., `Node` objects) are stored,
- **Index stores**: where index metadata are stored,
- **Vector stores**: where embedding vectors are stored.

The Document/Index stores rely on a common Key-Value store abstraction, which is also detailed below.


.. image:: ../_static/storage/storage.png


.. toctree::
   :maxdepth: 1
   :caption: Storage

   storage/save_load.md
   storage/customization.md
   storage/docstores.md
   storage/index_stores.md
   storage/vector_stores.md
   storage/kv_stores.md