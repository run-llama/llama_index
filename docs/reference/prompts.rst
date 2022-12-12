.. _Prompt-Templates:

Prompt Templates
=================

These are the reference prompt templates. 
We then document all prompts, with their required variables.

We then show the base prompt class, 
derived from `Langchain <https://langchain.readthedocs.io/en/latest/modules/prompt.html>`_.


**Summarization Prompt**

- Prompt to summarize the provided `text`.
- input variables: `["text"]`

**Tree Insert Prompt**

- Prompt to insert a new chunk of text `new_chunk_text` into the tree index. More specifically,
    this prompt has the LLM select the relevant candidate child node to continue tree traversal.
- input variables: `["num_chunks", "context_list", "new_chunk_text"]`

**Question-Answer Prompt**

- Prompt to answer a question `query_str` given a context `context_str`.
- input variables: `["context_str", "query_str"]`

**Refinement Prompt**

- Prompt to refine an existing answer `existing_answer` given a context `context_msg`,
    and a query `query_str`.
- input variables: `["query_str", "existing_answer", "context_msg"]`

**Keyword Extraction Prompt**

- Prompt to extract keywords from a text `text` with a maximum of `max_keywords` keywords.
- input variables: `["text", "max_keywords"]`

**Query Keyword Extraction Prompt**

- Prompt to extract keywords from a query `query_str` with a maximum of `max_keywords` keywords.
- input variables: `["question", "max_keywords"]`


**Tree Select Query Prompt**

- Prompt to select a candidate child node out of all child nodes provided in `context_list`,
    given a query `query_str`. `num_chunks` is the number of child nodes in `context_list`.

- input variables: `["num_chunks", "context_list", "query_str"]`


**Tree Select Query Prompt (Multiple)**

- Prompt to select multiple candidate child nodes out of all child nodes provided in `context_list`,
    given a query `query_str`. `branching_factor` refers to the number of child nodes to select, and
    `num_chunks` is the number of child nodes in `context_list`.

- input variables: `["num_chunks", "context_list", "query_str", "branching_factor"]`


**Base Prompt Class**

.. automodule:: gpt_index.prompts
   :members:
   :inherited-members:
   :exclude-members: Config, construct, copy, dict, from_examples, from_file, get_full_format_args, output_parser, save, template, template_format, update_forward_refs, validate_variable_names, json, template_is_valid


