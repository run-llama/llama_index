
.. _Ref-LLMs:


LLMs
====

A large language model (LLM) is a reasoning engine that can complete text,
chat with users, and follow instructions. 

LLM Implementations
^^^^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 1
   :caption: LLM Implementations

   llms/openai.rst
   llms/azure_openai.rst
   llms/huggingface.rst
   llms/langchain.rst
   llms/anthropic.rst
   llms/llama_cpp.rst
   llms/palm.rst
   llms/predibase.rst
   llms/replicate.rst
   llms/xinference.rst

LLM Interface
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: llama_index.llms.base.LLM
   :members:
   :inherited-members:

Schemas
^^^^^^^

.. autoclass:: llama_index.llms.base.MessageRole
   :members:
   :inherited-members:

.. autopydantic_model:: llama_index.llms.base.ChatMessage

.. autopydantic_model:: llama_index.llms.base.ChatResponse

.. autopydantic_model:: llama_index.llms.base.CompletionResponse

.. autopydantic_model:: llama_index.llms.base.LLMMetadata