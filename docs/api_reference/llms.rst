
.. _Ref-LLMs:


LLMs
====

A large language model (LLM) is a reasoning engine that can complete text,
chat with users, and follow instructions.

LLM Interface
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: llama_index.core.llms.base.LLM
   :members:
   :inherited-members:

Schemas
^^^^^^^

.. autoclass:: llama_index.core.llms.base.MessageRole
   :members:
   :inherited-members:

.. autopydantic_model:: llama_index.core.llms.base.ChatMessage

.. autopydantic_model:: llama_index.core.llms.base.ChatResponse

.. autopydantic_model:: llama_index.core.llms.base.CompletionResponse

.. autopydantic_model:: llama_index.core.llms.base.LLMMetadata
