# Fusion Retriever Packs

## Hybrid Fusion Pack

This LlamaPack provides an example of our hybrid fusion retriever method.

This specific template fuses results from our vector retriever and bm25 retriever; of course, you can provide any template you want.

Check out the [notebook here](https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_packs/fusion_retriever/hybrid_fusion/hybrid_fusion.ipynb).

### CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack HybridFusionRetrieverPack --download-dir ./hybrid_fusion_pack
```

You can then inspect the files at `./hybrid_fusion_pack` and use them as a template for your own project.

### Code Usage

You can download the pack to a the `./hybrid_fusion_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
HybridFusionRetrieverPack = download_llama_pack(
    "HybridFusionRetrieverPack", "./hybrid_fusion_pack"
)
```

From here, you can use the pack, or inspect and modify the pack in `./hybrid_fusion_pack`.

Then, you can set up the pack like so:

```python
# create the pack
hybrid_fusion_pack = HybridFusionRetrieverPack(
    nodes, chunk_size=256, vector_similarity_top_k=2, bm25_similarity_top_k=2
)
```

The `run()` function is a light wrapper around `query_engine.query()`.

```python
response = hybrid_fusion_pack.run("Tell me about a Music celebrity.")
```

You can also use modules individually.

```python
# use the fusion retriever
nodes = hybrid_fusion_pack.fusion_retriever.retrieve("query_str")

# use the vector retriever
nodes = hybrid_fusion_pack.vector_retriever.retrieve("query_str")
# use the bm25 retriever
nodes = hybrid_fusion_pack.bm25_retriever.retrieve("query_str")

# get the query engine
query_engine = hybrid_fusion_pack.query_engine
```

## Query Rewriting Retriever Pack

This LlamaPack provides an example of query rewriting through our fusion retriever.

This specific template takes in a single retriever, and generates multiple queries against the retriever, and then fuses the results together.

Check out the [notebook here](https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_packs/fusion_retriever/query_rewrite/query_rewrite.ipynb).

### CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack QueryRewritingRetrieverPack --download-dir ./query_rewriting_pack
```

You can then inspect the files at `./query_rewriting_pack` and use them as a template for your own project.

### Code Usage

You can download the pack to a the `./query_rewriting_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
QueryRewritingRetrieverPack = download_llama_pack(
    "QueryRewritingRetrieverPack", "./query_rewriting_pack"
)
```

From here, you can use the pack, or inspect and modify the pack in `./query_rewriting_pack`.

Then, you can set up the pack like so:

```python
# create the pack
query_rewriting_pack = QueryRewritingRetrieverPack(
    nodes,
    chunk_size=256,
    vector_similarity_top_k=2,
)
```

The `run()` function is a light wrapper around `query_engine.query()`.

```python
response = query_rewriting_pack.run("Tell me a bout a Music celebrity.")
```

You can also use modules individually.

```python
# use the fusion retriever
nodes = query_rewriting_pack.fusion_retriever.retrieve("query_str")

# use the vector retriever
nodes = query_rewriting_pack.vector_retriever.retrieve("query_str")

# get the query engine
query_engine = query_rewriting_pack.query_engine
```
