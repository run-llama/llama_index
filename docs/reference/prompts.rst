.. _Prompt-Templates:

Prompt Templates
=================

These are the reference prompt templates. 
We first document all core prompts, with their required variables.


We then show the base prompt class, 
derived from `Langchain <https://langchain.readthedocs.io/en/latest/modules/prompt.html>`_.

We also show links to default prompts.


Prompts
^^^^^^^

.. automodule:: gpt_index.prompts.prompts
   :members:
   :inherited-members:
   :exclude-members: get_full_format_args



Base Prompt Class
^^^^^^^^^^^^^^^^^

.. automodule:: gpt_index.prompts
   :members:
   :inherited-members:
   :exclude-members: Config, construct, copy, dict, from_examples, from_file, get_full_format_args, output_parser, save, template, template_format, update_forward_refs, validate_variable_names, json, template_is_valid


Default Prompts
^^^^^^^^^^^^^^^^^

The list of default prompts can be `found here <https://github.com/jerryjliu/gpt_index/blob/main/gpt_index/prompts/default_prompts.py>`_.

**NOTE**: we've also curated a set of refine prompts for ChatGPT use cases. 
The list of ChatGPT refine prompts can be 
`found here <https://github.com/jerryjliu/gpt_index/blob/main/gpt_index/prompts/chat_prompts.py>`_.