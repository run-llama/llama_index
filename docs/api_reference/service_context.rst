.. _Ref-Service-Context:

Service Context
=================

The service context container is a utility container for LlamaIndex
index and query classes. The container contains the following 
objects that are commonly used for configuring every index and
query, such as the LLMPredictor (for configuring the LLM),
the PromptHelper (for configuring input size/chunk size),
the BaseEmbedding (for configuring the embedding model), and more.

| 

.. toctree::
   :maxdepth: 1
   :caption: Service Context Classes

   service_context/embeddings.rst
   service_context/node_parser.rst
   service_context/prompt_helper.rst
   llms.rst

------------

.. automodule:: llama_index.indices.service_context
   :members:
   :inherited-members:
