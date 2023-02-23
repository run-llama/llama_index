.. _Ref-Indices-VectorStore:

Vector Store Index
==================

Below we show the vector store index class; it is the main
interface for specifying a vector store index. To specify a vector store 
index, you must also specify a Vector Store, shown below. 

We still maintain the some vector store index class for backwards compatibility.
However, these classes are deprecated and will be removed in a future release.

.. toctree::
   :maxdepth: 1

   vector_stores/stores.rst
   vector_stores/old_indices.rst

.. automodule:: gpt_index.indices.vector_store.base
   :members:
   :inherited-members:
   :exclude-members: delete, docstore, index_struct, index_struct_cls

