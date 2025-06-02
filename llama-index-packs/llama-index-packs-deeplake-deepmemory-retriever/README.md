# DeepLake DeepMemory Pack

This LlamaPack inserts your data into deeplake and instantiates a [deepmemory](https://docs.activeloop.ai/performance-features/deep-memory) retriever, which will use deepmemory during runtime to increase RAG's retrieval accuracy (recall).

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack DeepMemoryRetrieverPack --download-dir ./deepmemory_pack
```

You can then inspect the files at `./deepmemory_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to a `./deepmemory_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
DeepMemoryRetriever = download_llama_pack(
    "DeepMemoryRetrieverPack", "./deepmemory_pack"
)
```

From here, you can use the pack, or inspect and modify the pack in `./deepmemory_pack`.

Then, you can set up the pack like so:

```python
# setup pack arguments
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo

nodes = [...]

# create the pack
deepmemory_pack = DeepMemoryRetriever(
    dataset_path="llama_index",
    overwrite=False,
    nodes=nodes,
)
```

The `run()` function is a light wrapper around `query_engine.query()`.

```python
response = deepmemory_pack.run("Tell me a bout a Music celebritiy.")
```

You can also use modules individually.

```python
# use the retriever
retriever = deepmemory_pack.retriever
nodes = retriever.retrieve("query_str")

# use the query engine
query_engine = deepmemory_pack.query_engine
response = query_engine.query("query_str")
```
