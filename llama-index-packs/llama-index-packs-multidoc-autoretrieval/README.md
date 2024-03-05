# Multi-Document AutoRetrieval (with Weaviate) Pack

This LlamaPack implements structured hierarchical retrieval over multiple documents, using multiple @weaviate_io collections.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack MultiDocAutoRetrieverPack --download-dir ./multidoc_autoretrieval_pack
```

You can then inspect the files at `./multidoc_autoretrieval_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to a the `./multidoc_autoretrieval_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
MultiDocAutoRetrieverPack = download_llama_pack(
    "MultiDocAutoRetrieverPack", "./multidoc_autoretrieval_pack"
)
```

From here, you can use the pack. To initialize it, you need to define a few arguments, see below.

Then, you can set up the pack like so:

```python
# setup pack arguments
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo

import weaviate

# cloud
auth_config = weaviate.AuthApiKey(api_key="<api_key>")
client = weaviate.Client(
    "https://<cluster>.weaviate.network",
    auth_client_secret=auth_config,
)

vector_store_info = VectorStoreInfo(
    content_info="Github Issues",
    metadata_info=[
        MetadataInfo(
            name="state",
            description="Whether the issue is `open` or `closed`",
            type="string",
        ),
        ...,
    ],
)

# metadata_nodes is set of nodes with metadata representing each document
# docs is the source docs
# metadata_nodes and docs must be the same length
metadata_nodes = [TextNode(..., metadata={...}), ...]
docs = [Document(...), ...]

pack = MultiDocAutoRetrieverPack(
    client,
    "<metadata_index_name>",
    "<doc_chunks_index_name>",
    metadata_nodes,
    docs,
    vector_store_info,
    auto_retriever_kwargs={
        # any kwargs for the auto-retriever
        ...
    },
)
```

The `run()` function is a light wrapper around `query_engine.query()`.

```python
response = pack.run("Tell me a bout a Music celebritiy.")
```

You can also use modules individually.

```python
# use the retriever
retriever = pack.retriever
nodes = retriever.retrieve("query_str")

# use the query engine
query_engine = pack.query_engine
response = query_engine.query("query_str")
```
