Query Interface
===============

LlamaIndex provides a *query interface* over your index or graph structure. This query interface
allows you to both retrieve the set of relevant documents, as well as synthesize a response.

- The basic query interface is found in our usage pattern guide. The guide
  details how to specify parameters for a basic query over a single index structure.
- A more advanced query interface is found in our composability guide. The guide
  describes how to specify a graph over multiple index structures.
- Finally, we provide a guide to our **Query Transformations** module. 


.. toctree::
   :maxdepth: 1
   :caption: Query Interface

   /guides/primer/usage_pattern.md
   /how_to/index_structs/composability.rst
   /how_to/query/query_transformations.md
   .. provide query transformations guide