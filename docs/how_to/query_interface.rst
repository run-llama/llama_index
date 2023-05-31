🔍 Query Interface
===============


Querying an index or a graph involves a three main components:

- **Retrievers**: A retriever class retrieves a set of Nodes from an index given a query.
- **Response Synthesizer**: This class takes in a set of Nodes and synthesizes an answer given a query.
- **Query Engine**: This class takes in a query and returns a Response object. It can make use
   of Retrievers and Response Synthesizer modules under the hood.

.. image:: /_static/query/query_classes.png
  :width: 600


Design Philosophy: Progressive Disclosure of Complexity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Progressive disclosure of complexity is a design philosophy that aims to strike 
ka balance between the needs of beginners and experts. The idea is that you should 
give users the simplest and most straightforward interface or experience possible 
when they first encounter a system or product, but then gradually reveal more 
complexity and advanced features as users become more familiar with the system. 
This can help prevent users from feeling overwhelmed or intimidated by a system 
that seems too complex, while still giving experienced users the tools they need 
to accomplish advanced tasks.

.. image:: /_static/query/disclosure.png
  :width: 600


In the case of LlamaIndex, we've tried to balance simplicity and complexity by 
providing a high-level API that's easy to use out of the box, but also a low-level 
composition API that gives experienced users the control they need to customize the 
system to their needs. By doing this, we hope to make LlamaIndex accessible to 
beginners while still providing the flexibility and power that experienced users need.

Resources
^^^^^^^^^

- The basic query interface over an index is found in our usage pattern guide. The guide
  details how to specify parameters for a retriever/synthesizer/query engine over a 
  single index structure.
- A more advanced query interface is found in our composability guide. The guide
  describes how to specify a graph over multiple index structures.
- We also provide a guide to some of our more advanced components, which can be added 
  to a retriever or a query engine. See our **Query Transformations** and 
  **Node Postprocessor** modules. 


.. toctree::
   :maxdepth: 1
   :caption: Guides

   /guides/primer/usage_pattern.md
   /how_to/index_structs/composability.rst
   /how_to/query/query_transformations.md
   /how_to/query/second_stage.md
   /how_to/query/optimizers.md
   /how_to/query/response_synthesis.md
   /how_to/query/query_engines.md
   /how_to/query/chat_engines.md