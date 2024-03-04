# Vanna AI LLamaPack

Vanna AI is an open-source RAG framework for SQL generation. It works in two steps:

1. Train a RAG model on your data
2. Ask questions (use reference corpus to generate SQL queries that can run on your db).

Check out the [Github project](https://github.com/vanna-ai/vanna) and the [docs](https://vanna.ai/docs/) for more details.

This LlamaPack creates a simple `VannaQueryEngine` with vanna, ChromaDB and OpenAI, and allows you to train and ask questions over a SQL database.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack VannaPack --download-dir ./vanna_pack
```

You can then inspect the files at `./vanna_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to a `./vanna_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
VannaPack = download_llama_pack("VannaPack", "./vanna_pack")
```

From here, you can use the pack, or inspect and modify the pack in `./vanna_pack`.

Then, you can set up the pack like so:

```python
pack = VannaPack(
    openai_api_key="<openai_api_key>",
    sql_db_url="chinook.db",
    openai_model="gpt-3.5-turbo",
)
```

The `run()` function is a light wrapper around `llm.complete()`.

```python
response = pack.run("List some sample albums")
```

You can also use modules individually.

```python
query_engine = pack.get_modules()["vanna_query_engine"]
```
