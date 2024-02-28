# Zephyr Query Engine Pack

Create a query engine using completely local and private models -- `HuggingFaceH4/zephyr-7b-beta` for the LLM and `BAAI/bge-base-en-v1.5` for embeddings.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack ZephyrQueryEnginePack --download-dir ./zephyr_pack
```

You can then inspect the files at `./zephyr_pack` and use them as a template for your own project.

## Code Usage

You can download the pack to a the `./zephyr_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
ZephyrQueryEnginePack = download_llama_pack(
    "ZephyrQueryEnginePack", "./zephyr_pack"
)

# You can use any llama-hub loader to get documents!
zephyr_pack = ZephyrQueryEnginePack(documents)
```

From here, you can use the pack, or inspect and modify the pack in `./zephyr_pack`.

The `run()` function is a light wrapper around `index.as_query_engine().query()`.

```python
response = zephyr_pack.run(
    "What did the author do growing up?", similarity_top_k=2
)
```

You can also use modules individually.

```python
# Use the llm
llm = zephyr_pack.llm
response = llm.complete("What is HuggingFace?")

# Use the index directly
index = zephyr_pack.index
query_engine = index.as_query_engine()
retriever = index.as_retriever()
```
