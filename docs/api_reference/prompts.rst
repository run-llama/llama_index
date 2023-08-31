.. _Prompt-Templates:

Prompt Templates
=================

These are the reference prompt templates. 

We first show links to default prompts.

We then show the base prompt template class and its subclasses.

Default Prompts
^^^^^^^^^^^^^^^^^


* `Completion prompt templates <https://github.com/jerryjliu/llama_index/blob/main/llama_index/prompts/default_prompts.py>`_.
* `Chat prompt templates <https://github.com/jerryjliu/llama_index/blob/main/llama_index/prompts/chat_prompts.py>`_.
* `Selector prompt templates <https://github.com/jerryjliu/llama_index/blob/main/llama_index/prompts/default_prompt_selectors.py>`_.



Prompt Classes
^^^^^^^^^^^^^^^^^

.. autopydantic_model:: llama_index.prompts.base.BasePromptTemplate

.. autopydantic_model:: llama_index.prompts.base.PromptTemplate

.. autopydantic_model:: llama_index.prompts.base.ChatPromptTemplate

.. autopydantic_model:: llama_index.prompts.base.SelectorPromptTemplate

.. autopydantic_model:: llama_index.prompts.base.LangchainPromptTemplate


Subclass Prompts (deprecated)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Deprecated, but still available for reference at `this link <https://github.com/jerryjliu/llama_index/blob/113109365b216428440b19eb23c9fae749d6880a/llama_index/prompts/prompts.py>`_.
