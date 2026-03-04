# Recursive Retriever Packs

## Embedded Tables Retriever Pack w/ Unstructured.io

This LlamaPack provides an example of our embedded tables retriever.

This specific template shows the e2e process of building this. It loads
a document, builds a hierarchical node graph (with bigger parent nodes and smaller
child nodes).

Check out the [notebook here](https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_packs/recursive_retriever/embedded_tables_unstructured/embedded_tables.ipynb).

### CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack EmbeddedTablesUnstructuredRetrieverPack --download-dir ./embedded_tables_unstructured_pack
```

You can then inspect the files at `./embedded_tables_unstructured_pack` and use them as a template for your own project.

### Code Usage

You can download the pack to a the `./embedded_tables_unstructured_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
EmbeddedTablesUnstructuredRetrieverPack = download_llama_pack(
    "EmbeddedTablesUnstructuredRetrieverPack",
    "./embedded_tables_unstructured_pack",
)
```

From here, you can use the pack, or inspect and modify the pack in `./embedded_tables_unstructured_pack`.

Then, you can set up the pack like so:

```python
# create the pack
# get documents from any data loader
embedded_tables_unstructured_pack = EmbeddedTablesUnstructuredRetrieverPack(
    "tesla_2021_10k.htm",
)
```

The `run()` function is a light wrapper around `query_engine.query()`.

```python
response = embedded_tables_unstructured_pack.run(
    "What was the revenue in 2020?"
)
```

You can also use modules individually.

```python
# get the node parser
node_parser = embedded_tables_unstructured_pack.node_parser

# get the retriever
retriever = embedded_tables_unstructured_pack.recursive_retriever

# get the query engine
query_engine = embedded_tables_unstructured_pack.query_engine
```

## Recursive Retriever - Small-to-big retrieval

This LlamaPack provides an example of our recursive retriever (small-to-big).

This specific template shows the e2e process of building this. It loads
a document, builds a hierarchical node graph (with bigger parent nodes and smaller
child nodes).

Check out the [notebook here](https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_packs/recursive_retriever/small_to_big/small_to_big.ipynb).

### CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack RecursiveRetrieverSmallToBigPack --download-dir ./recursive_retriever_stb_pack
```

You can then inspect the files at `./recursive_retriever_stb_pack` and use them as a template for your own project.

### Code Usage

You can download the pack to a the `./recursive_retriever_stb_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
RecursiveRetrieverSmallToBigPack = download_llama_pack(
    "RecursiveRetrieverSmallToBigPack", "./recursive_retriever_stb_pack"
)
```

From here, you can use the pack, or inspect and modify the pack in `./recursive_retriever_stb_pack`.

Then, you can set up the pack like so:

```python
# create the pack
# get documents from any data loader
recursive_retriever_stb_pack = RecursiveRetrieverSmallToBigPack(
    documents,
)
```

The `run()` function is a light wrapper around `query_engine.query()`.

```python
response = recursive_retriever_stb_pack.run(
    "Tell me a bout a Music celebrity."
)
```

You can also use modules individually.

```python
# get the recursive retriever
recursive_retriever = recursive_retriever_stb_pack.recursive_retriever

# get the query engine
query_engine = recursive_retriever_stb_pack.query_engine
```
