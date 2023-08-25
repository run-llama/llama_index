.. _Ref-Query:

Querying an Index
=================

This doc shows the classes that are used to query indices.


Main Query Classes
^^^^^^^^^^^^^^^^^^

Querying an index involves a three main components:

- **Retrievers**: A retriever class retrieves a set of Nodes from an index given a query.
- **Response Synthesizer**: This class takes in a set of Nodes and synthesizes an answer given a query.
- **Query Engine**: This class takes in a query and returns a Response object. It can make use of Retrievers and Response Synthesizer modules under the hood.
- **Chat Engines**: This class enables conversation over a knowledge base. It is the stateful version of a query engine that keeps track of conversation history.


.. toctree::
   :maxdepth: 1
   :caption: Main query classes

   query/retrievers.rst
   query/response_synthesizer.rst
   query/query_engines.rst
   query/chat_engines.rst


Additional Query Classes
^^^^^^^^^^^^^^^^^^^^^^^^

We also detail some additional query classes below.

- **Query Bundle**: This is the input to the query classes: retriever, response synthesizer, 
   and query engine. It enables the user to customize the string(s) 
   used for embedding-based query.
- **Query Transform**: This class augments a raw query string with 
   associated transformations to improve index querying. Can be used 
   with a Retriever (see TransformRetriever) or QueryEngine.


.. toctree::
   :maxdepth: 1
   :caption: Additional query classes

   query/query_bundle.rst
   query/query_transform.rst

