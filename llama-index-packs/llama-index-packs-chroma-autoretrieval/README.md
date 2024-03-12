# Chroma AutoRetrieval Pack

This LlamaPack inserts your data into chroma and instantiates an auto-retriever, which will use the LLM at runtime to set metadata filtering, top-k, and query string.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack ChromaAutoretrievalPack --download-dir ./chroma_pack
```

You can then inspect the files at `./chroma_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to a the `./chroma_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
ChromaAutoretrievalPack = download_llama_pack(
    "ChromaAutoretrievalPack", "./chroma_pack"
)
```

From here, you can use the pack, or inspect and modify the pack in `./chroma_pack`.

Then, you can set up the pack like so:

```python
# setup pack arguments
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo

vector_store_info = VectorStoreInfo(
    content_info="brief biography of celebrities",
    metadata_info=[
        MetadataInfo(
            name="category",
            type="str",
            description=(
                "Category of the celebrity, one of [Sports Entertainment, Business, Music]"
            ),
        ),
    ],
)

import chromadb

client = chromadb.EphemeralClient()

nodes = [...]

# create the pack
chroma_pack = ChromaAutoretrievalPack(
    collection_name="test",
    vector_store_info=vector_store_index,
    nodes=nodes,
    client=client,
)
```

The `run()` function is a light wrapper around `query_engine.query()`.

```python
response = chroma_pack.run("Tell me a bout a Music celebritiy.")
```

You can also use modules individually.

```python
# use the retriever
retriever = chroma_pack.retriever
nodes = retriever.retrieve("query_str")

# use the query engine
query_engine = chroma_pack.query_engine
response = query_engine.query("query_str")
```
