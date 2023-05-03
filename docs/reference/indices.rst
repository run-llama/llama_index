.. _Ref-Indices:

Indices
=======

This doc shows both the overarching class used to represent an index. These
classes allow for index creation, insertion, and also querying.
We first show the different index subclasses.
We then show the base class that all indices inherit from, which contains 
parameters and methods common to all indices.

|

.. toctree::
   :maxdepth: 1
   :caption: Index Data Structures

   indices/list.rst
   indices/table.rst
   indices/tree.rst
   indices/vector_store.rst
   indices/struct_store.rst
   indices/kg.rst
   indices/empty.rst


Base Index Class
^^^^^^^^^^^^^^^^

.. automodule:: llama_index.indices.base
   :members:
   :inherited-members:
