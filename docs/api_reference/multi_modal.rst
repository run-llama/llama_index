
.. _Ref-Multi-Modal-LLMs:


Multi-Modal LLMs, Vector Stores, Embeddings, Retriever, and Query Engine
====

Multi-Modal large language model (LLM) is a Multi-Modal reasoning engine that
can complete text and image chat with users, and follow instructions.

Multi-Modal LLM Implementations
^^^^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 1
   :caption: Multi-Modal LLM Implementations

   multi_modal/openai.rst
   multi_modal/replicate.rst

Multi-Modal LLM Interface
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: llama_index.multi_modal_llms.base.MultiModalLLM
   :members:
   :inherited-members:

Multi-Modal Embedding
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: llama_index.embeddings.multi_modal_base.MultiModalEmbedding
   :members:
   :inherited-members:

Multi-Modal Vector Store Index
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: llama_index.indices.multi_modal.base.MultiModalVectorStoreIndex
   :members:
   :inherited-members:

Multi-Modal Vector Index Retriever
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: llama_index.indices.multi_modal.retriever.MultiModalVectorIndexRetriever
   :members:
   :inherited-members:

Multi-Modal Retriever Interface
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: llama_index.core.base_multi_modal_retriever.MultiModalRetriever
   :members:
   :inherited-members:


Multi-Modal Simple Query Engine
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: llama_index.query_engine.multi_modal.SimpleMultiModalQueryEngine
   :members:
   :inherited-members:
