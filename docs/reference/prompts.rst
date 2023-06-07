.. _Prompt-Templates:

Prompt Templates
=================

These are the reference prompt templates. 

We first show links to default prompts.

We then show the base prompt class, 
derived from `Langchain <https://langchain.readthedocs.io/en/latest/modules/prompt.html>`_.

Default Prompts
^^^^^^^^^^^^^^^^^

The list of default prompts can be `found here <https://github.com/jerryjliu/llama_index/blob/main/llama_index/prompts/default_prompts.py>`_.

**NOTE**: we've also curated a set of refine prompts for ChatGPT use cases. 
The list of ChatGPT refine prompts can be 
`found here <https://github.com/jerryjliu/llama_index/blob/main/llama_index/prompts/chat_prompts.py>`_.


Prompts
^^^^^^^

.. automodule:: llama_index.prompts.prompts
   :members:
   :inherited-members:
   :exclude-members: get_full_format_args


Base Prompt Class
^^^^^^^^^^^^^^^^^

.. automodule:: llama_index.prompts
   :members:
   :inherited-members:
   :exclude-members: Config, construct, copy, dict, from_examples, from_file, get_full_format_args, output_parser, save, template, template_format, update_forward_refs, validate_variable_names, json, template_is_valid

