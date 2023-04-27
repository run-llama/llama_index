.. _Ref-Storage:


Storage
=================

LlamaIndex offers core abstractions around storage of Nodes, indices, and vectors.
A key abstraction is the `StorageContext` - this contains the underlying
`BaseDocumentStore` (for nodes), `BaseIndexStore` (for indices), and `VectorStore` (for vectors).

See below pages for more details.

.. toctree::
   :maxdepth: 1
   :caption: Storage Classes

   storage/docstore.rst
   storage/index_store.rst
   storage/vector_store.rst

.. automodule:: gpt_index.storage.storage_context
   :members:
   :inherited-members: