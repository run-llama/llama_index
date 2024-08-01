# Ollama Query Engine Pack

Create a query engine using completely local by Ollama

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack OllamaQueryEnginePack --download-dir ./ollama_pack
```

You can then inspect the files at `./ollama_pack` and use them as a template for your own project.

## Code Usage

You can download the pack to a the `./ollama_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
OllamaQueryEnginePack = download_llama_pack(
    "OllamaQueryEnginePack", "./ollama_pack"
)

# You can use any llama-hub loader to get documents!
ollama_pack = OllamaQueryEnginePack(model="llama2", documents=documents)
```

From here, you can use the pack, or inspect and modify the pack in `./ollama_pack`.

The `run()` function is a light wrapper around `index.as_query_engine().query()`.

```python
response = ollama_pack.run("What is the title of the book of John?")
```

You can also use modules individually.

```python
# Use the llm
llm = ollama_pack.llm
response = llm.complete("What is Ollama?")

# Use the index directly
index = ollama_pack.index
query_engine = index.as_query_engine()
retriever = index.as_retriever()
```
