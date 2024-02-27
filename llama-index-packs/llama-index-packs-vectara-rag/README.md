# Vectara RAG Pack

This LlamaPack provides an end-to-end Retrieval Augmented Generation flow using Vectara.

Before you start, if you have not done so already, you would need to follow these steps:

- Create a [free Vectara account](https://vectara.com/integrations/llamaindex).
- Create a [corpus](https://docs.vectara.com/docs/console-ui/creating-a-corpus) to store your data
- Create an [API key](https://docs.vectara.com/docs/common-use-cases/app-authn-authz/api-keys) with QueryService and IndexService access so you can access this corpus

You can configure your `.env` file or provide these arguments directly when creating your VectaraIndex:

```
VECTARA_CUSTOMER_ID=your_customer_id
VECTARA_CORPUS_ID=your_corpus_id
VECTARA_API_KEY=your-vectara-api-key
```

## CLI Usage

You can download the Vectara llamapack directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack VectaraRagPack --download-dir ./vectara_rag_pack
```

Feel free to inspect the files at `./vectara_rag_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to a `./vectara_rag_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

VectaraRAG = download_llama_pack("VectaraRagPack", "./vectara_rag_pack")
```

Then, you can set up the pack in two ways:

1. If you want to ingest documents into the Vectara Corpus:

```python
nodes = [...]
vectara = VectaraRAG(nodes=nodes)
```

2. If you already indexed data on Vectara, and just want to use the retrieval/query functionality:

```python
vectara = VectaraRAG()
```

Additional optional arguments to VectaraRAG:

- `similarity_top_k`: determines the number of results to return. Defaults to 5.
- `n_sentences_before` and `n_sentences_after`: determine the number of sentences before/after the
  matching fact to use with the summarization LLM. defaults to 2.
- `vectara_query_mode`: 'default' or 'mmr'
- `summary_enabled`: whether to generate summaries or not. Defaults to True.
- When summary_enabled is True, you can set the following:
  - `summary_response_lang`: language to use (ISO 639-2 code) for summary generation. defaults to "eng".
  - `summary_num_results`: number of results to use for summary generation. Defaults to 7.
  - `summary_prompt_name`: name of the prompt to use for summary generation.
    Defaults to 'vectara-summary-ext-v1.2.0'.
    Scale customers can use 'vectara-summary-ext-v1.3.0
- when `vectara_query_mode` is "mmr", you can set the following to control MMR:
  - `mmr_k`: number of results to fetch for MMR, defaults to 50
  - `mmr_diversity_bias`: number between 0 and 1 that determines the degree
    of diversity among the results with 0 corresponding
    to minimum diversity and 1 to maximum diversity. Defaults to 0.3.

For example to use maximal diversity with MMR:

```python
vectara = VectaraRAG(
    vectara_query_mode="mmr", mmr_k=50, mmr_diversity_bias=1.0
)
```

Or if you want to include more results in the Vectara generated summarization you can try:

```python
vectara = VectaraRAG(summary_num_results=12)
```

Once you have the Vectara RAG object, you can now use it as a retriever:

```python
# use the retriever
nodes = vectara.retrieve("Is light a wave or a particle?")
```

Or as a query engine (with Vectara summarization call):

```python
# use the query engine
response = vectara._query_engine.query(
    "Is light a wave or a particle?"
).response
```

Note that the `run()` function is a light wrapper around `query_engine.query()`.

```python
response = vectara.run("Is light a wave or a particle?").response
```

Enjoy your Vectara RAG pack!
