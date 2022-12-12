.. _Prompt-Templates:

Prompt Templates
=================

These are the reference prompt templates. 
We then document all prompts, with their required variables.

We then show the base prompt class, 
derived from [Langchain](https://langchain.readthedocs.io/en/latest/modules/prompt.html).


**Summarization Prompt**

- input variables: `["text"]`

**Insert Prompt**

- input variables: `["num_chunks", "context_list", "new_chunk_text"]`

**Question-Answer Prompt**

- input variables: `["context_str", "query_str"]`



.. automodule:: gpt_index.prompts
   :members:
   :inherited-members:
   :exclude-members: Config, construct, copy, dict, from_examples, from_file, get_full_format_args, output_parser, save, template, template_format, update_forward_refs, validate_variable_names, json, template_is_valid


