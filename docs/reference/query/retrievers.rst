.. _Ref-Retrievers:

Retrievers
=================

Index Retrievers
^^^^^^^^^^^^^^^^
Below we show index-specific retriever classes.

.. toctree::
   :maxdepth: 1
   :caption: Index Retrievers

   retrievers/empty.rst
   retrievers/kg.rst
   retrievers/list.rst
   retrievers/table.rst
   retrievers/tree.rst
   retrievers/vector_store.rst


Additional Retrievers
^^^^^^^^^^^^^^^^^^^^^

Here we show additional retriever classes; these classes
can augment existing retrievers with new capabilities (e.g. query transforms).

.. toctree::
   :maxdepth: 1
   :caption: Additional Retrievers

   retrievers/transform.rst


Base Retriever
^^^^^^^^^^^^^^^^^^^^^

Here we show the base retriever class, which contains the `retrieve`
method which is shared amongst all retrievers.


.. automodule:: gpt_index.indices.base_retriever
   :members:
   :inherited-members:
..    :exclude-members: index_struct, query, set_llm_predictor, set_prompt_helper
