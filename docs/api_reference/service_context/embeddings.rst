.. _Ref-Embeddings:

Embeddings
=================

Users have a few options to choose from when it comes to embeddings.

- :code:`OpenAIEmbedding`: the default embedding class. Defaults to "text-embedding-ada-002"
- :code:`LangchainEmbedding`: a wrapper around Langchain's embedding models.


.. automodule:: llama_index.embeddings
   :members:
   :inherited-members:
   :exclude-members: OAEM, OpenAIEmbeddingMode


.. .. automodule:: llama_index.embeddings.openai
..    :members:
..    :inherited-members:
..    :exclude-members: OAEM, OpenAIEmbeddingMode


.. We also introduce a :code:`LangchainEmbedding` class, which is a wrapper around Langchain's embedding models.
.. A full list of embeddings can be found `here <https://langchain.readthedocs.io/en/latest/reference/modules/embeddings.html>`_.

.. .. automodule:: llama_index.embeddings.langchain
..    :members:
..    :inherited-members:

