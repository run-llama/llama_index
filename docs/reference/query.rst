.. _Ref-Query:

Querying an Index
=================

This doc specifically shows the classes that are used to query indices.
We first show index-specific query subclasses.
We then show the base query class, which contains parameters that are shared
among all queries. 

.. toctree::
   :maxdepth: 1
   :caption: Index-specific Query Subclasses

   indices/list_query.rst
   indices/table_query.rst
   indices/tree_query.rst
   indices/vector_store_query.rst


Base Query Class
^^^^^^^^^^^^^^^^

.. automodule:: gpt_index.indices.query.base
   :members:
   :inherited-members:
   :exclude-members: BaseQueryRunner