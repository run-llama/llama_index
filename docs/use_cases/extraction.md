# Structured Data Extraction

LLMs are capable of ingesting large amounts of unstructured data and returning it in structured formats, and LlamaIndex is set up to make this easy.

Using LlamaIndex, you can get an LLM to read natural language and identify semantically important details such as names, dates, addresses, and figures, and return them in a consistent structured format regardless of the source format.

This can be especially useful when you have unstructured source material like chat logs and conversation transcripts.

Once you have structured data you can send them to a database, or you can parse structured outputs in code to automate workflows.

Examples:

- [Extracting names and locations from descriptions of people](/docs/examples/output_parsing/df_program.ipynb)
- [Extracting album data from music reviews](/docs/examples/llm/llama_api.ipynb)
