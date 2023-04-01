Querying a Vector Store Index
=============================

We first show the base vector store query class. We then show the 
query classes specific to each vector store.

Base Vector Store Query Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: gpt_index.indices.vector_store.base_query
   :members:
   :inherited-members:
   :exclude-members: index_struct, query, set_llm_predictor, set_prompt_helper


Vector Store-specific Query Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: gpt_index.indices.vector_store.queries
   :members:
   :inherited-members:
   :exclude-members: index_struct, query, set_llm_predictor, set_prompt_helper