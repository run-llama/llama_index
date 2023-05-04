.. _Ref-Storage:


Storage Context
=================

LlamaIndex offers core abstractions around storage of Nodes, indices, and vectors.
A key abstraction is the `StorageContext` - this contains the underlying
`BaseDocumentStore` (for nodes), `BaseIndexStore` (for indices), and `VectorStore` (for vectors).


The Document/Node and index stores rely on a common `KVStore` abstraction, which is also detailed below.


We show the API references for the Storage Classes, loading indices from the Storage Context, and the Storage Context class itself below.

|

.. toctree::
   :maxdepth: 1
   :caption: Storage Classes

   storage/docstore.rst
   storage/index_store.rst
   storage/vector_store.rst
   storage/kv_store.rst

| 

.. toctree::
   :maxdepth: 1
   :caption: Loading Indices

   storage/indices_save_load.rst

------------

.. automodule:: llama_index.storage.storage_context
   :members:
   :inherited-members: