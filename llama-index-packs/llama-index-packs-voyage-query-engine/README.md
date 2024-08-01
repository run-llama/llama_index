# Voyage Query Engine Pack

Create a query engine using GPT4 and [Voyage AI](https://docs.voyageai.com/embeddings/) Embeddings.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack VoyageQueryEnginePack --download-dir ./voyage_pack
```

You can then inspect the files at `./voyage_pack` and use them as a template for your own project.

## Code Usage

You can download the pack to a the `./voyage_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
VoyageQueryEnginePack = download_llama_pack(
    "VoyageQueryEnginePack", "./voyage_pack"
)

# You can use any llama-hub loader to get documents!
voyage_pack = VoyageQueryEnginePack(documents)
```

From here, you can use the pack, or inspect and modify the pack in `./voyage_pack`.

The `run()` function is a light wrapper around `index.as_query_engine().query()`.

```python
response = voyage_pack.run(
    "What did the author do growing up?", similarity_top_k=2
)
```

You can also use modules individually.

```python
# Use the index directly
index = voyage_pack.index
query_engine = index.as_query_engine()
retriever = index.as_retriever()
```
