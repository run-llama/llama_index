# DeepLake DeepMemory Pack

This LlamaPack inserts your multimodal data (texts, images) into deeplake and instantiates an deeplake retriever, which will use clip for embedding images and GPT4-V during runtime.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack DeepLakeMultimodalRetrieverPack --download-dir ./deeplake_multimodal_pack
```

You can then inspect the files at `./deeplake_multimodal_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to a `./deeplake_multimodal_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
DeepLakeMultimodalRetriever = download_llama_pack(
    "DeepLakeMultimodalRetrieverPack", "./deeplake_multimodal_pack"
)
```

From here, you can use the pack, or inspect and modify the pack in `./deepmemory_pack`.

Then, you can set up the pack like so:

```python
# setup pack arguments
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo

# collection of image and text nodes
nodes = [...]

# create the pack
deeplake_pack = DeepLakeMultimodalRetriever(
    nodes=nodes, dataset_path="llama_index", overwrite=False
)
```

The `run()` function is a light wrapper around `SimpleMultiModalQueryEngine`.

```python
response = deeplake_pack.run("Tell me a bout a Music celebrity.")
```

You can also use modules individually.

```python
# use the retriever
retriever = deeplake_pack.retriever
nodes = retriever.retrieve("query_str")

# use the query engine
query_engine = deeplake_pack.query_engine
response = query_engine.query("query_str")
```
