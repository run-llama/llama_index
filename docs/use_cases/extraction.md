# Structured Data Extraction

LLMs are capable of ingesting large amounts of unstructured data and returning it in structured formats, and LlamaIndex is set up to make this easy.

Using LlamaIndex, you can get an LLM to read natural language and identify semantically important details such as names, dates, addresses, and figures, and return them in a consistent structured format regardless of the source format.

This can be especially useful when you have unstructured source material like chat logs and conversation transcripts.

Once you have structured data you can send them to a database, or you can parse structured outputs in code to automate workflows.

## Core Guides

Check out our Structured Output guide for a comprehensive overview of structured data extraction with LlamaIndex. Do it in a standalone fashion (Pydantic program) or as part of a RAG pipeline. We also have standalone output parsing modules that you can use yourself with an LLM / prompt.

```{toctree}
---
maxdepth: 2
---
/module_guides/querying/structured_outputs/structured_outputs.md
/module_guides/querying/structured_outputs/output_parser.md
```

We also have multi-modal structured data extraction. [Check it out](multi-modal-pydantic-program).

## Misc Examples

Some additional miscellaneous examples highlighting use cases:

```{toctree}
---
maxdepth: 1
---
Extracting names and locations from descriptions of people </examples/output_parsing/df_program.ipynb>
Extracting album data from music reviews </examples/llm/llama_api.ipynb>
```
