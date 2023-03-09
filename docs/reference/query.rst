.. _Ref-Query:

Querying an Index
=================

This doc shows the classes that are used to query indices.
We first show index-specific query subclasses.
We then show how to define a query config in order to recursively query
multiple indices that are `composed </how_to/composability.html>`_ together.
We then show the base query class, which contains parameters that are shared
among all queries. 
Lastly, we show how to customize the string(s) used for an embedding-based query.

.. toctree::
   :maxdepth: 1
   :caption: Index-specific Query Subclasses

   indices/list_query.rst
   indices/table_query.rst
   indices/tree_query.rst
   indices/vector_store_query.rst
   indices/struct_store_query.rst
   indices/kg_query.rst
   indices/empty_query.rst


This section shows how to define a query config in order to recursively query
multiple indices that are `composed </how_to/composability.html>`_ together.


.. toctree::
   :maxdepth: 1
   :caption: Query Configs for Composed Indices

   indices/composability_query.rst

Base Query Class
^^^^^^^^^^^^^^^^

.. automodule:: gpt_index.indices.query.base
   :members:
   :inherited-members:
   :exclude-members: BaseQueryRunner

Query Bundle
^^^^^^^^^^^^^^^^
Query bundle enables user to customize the string(s) used for embedding-based query.

.. automodule:: gpt_index.indices.query.schema
   :members: QueryBundle
   :inherited-members:
   :exclude-members:

Query Transform

Query transform augments a raw query string with associated transformations 
to improve index querying.

.. automodule:: gpt_index.indices.query.query_transform
   :members:
   :inherited-members:
   :exclude-members: