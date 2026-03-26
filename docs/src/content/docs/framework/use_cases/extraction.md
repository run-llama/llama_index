---
title: Structured Data Extraction
---

LLMs are capable of ingesting large amounts of unstructured data and returning it in structured formats, and LlamaIndex is set up to make this easy.

Using LlamaIndex, you can get an LLM to read natural language and identify semantically important details such as names, dates, addresses, and figures, and return them in a consistent structured format regardless of the source format.

This can be especially useful when you have unstructured source material like chat logs and conversation transcripts.

Once you have structured data you can send them to a database, or you can parse structured outputs in code to automate workflows.

## Full tutorial

Our Learn section has a [full tutorial on structured data extraction](/python/framework/understanding/extraction). We recommend starting out there.

There is also an [example notebook](/python/examples/structured_outputs/structured_outputs) demonstrating some of the techniques from the tutorial.

## Other Guides

For a more comprehensive overview of structured data extraction with LlamaIndex, including lower-level modules, check out the following guides:

- [Structured Outputs](/python/framework/module_guides/querying/structured_outputs)
- [Pydantic Programs](/python/framework/module_guides/querying/structured_outputs/pydantic_program)
- [Output Parsing](/python/framework/module_guides/querying/structured_outputs/output_parser)

We also have multi-modal structured data extraction. [Check it out](/python/framework/use_cases/multimodal#simple-evaluation-of-multi-modal-rag).

## Miscellaneous Examples

Some additional examples highlighting use cases:

- [Extracting names and locations from descriptions of people](/python/examples/output_parsing/df_program)
- [Extracting album data from music reviews](/python/examples/llm/llama_api)
- [Extracting information from emails](/python/examples/usecases/email_data_extraction)
