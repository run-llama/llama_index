# Auto Merging Retriever Pack

This LlamaPack provides an example of our auto-merging retriever.

This specific template shows the e2e process of building this. It loads
a document, builds a hierarchical node graph (with bigger parent nodes and smaller
child nodes).

Check out the [notebook here](https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_packs/auto_merging_retriever/auto_merging_retriever.ipynb).

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack AutoMergingRetrieverPack --download-dir ./auto_merging_retriever_pack
```

You can then inspect the files at `./auto_merging_retriever_pack` and use them as a template for your own project.

## Code Usage

You can download the pack to a the `./auto_merging_retriever_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
AutoMergingRetrieverPack = download_llama_pack(
    "AutoMergingRetrieverPack", "./auto_merging_retriever_pack"
)
```

From here, you can use the pack, or inspect and modify the pack in `./auto_merging_retriever_pack`.

Then, you can set up the pack like so:

```python
# create the pack
# get documents from any data loader
auto_merging_retriever_pack = AutoMergingRetrieverPack(
    documents,
)
```

The `run()` function is a light wrapper around `query_engine.query()`.

```python
response = auto_merging_retriever_pack.run(
    "Tell me about what the author did growing up."
)
```

You can also use modules individually.

```python
# get the node parser
node_parser = auto_merging_retriever_pack.node_parser

# get the retriever
retriever = auto_merging_retriever_pack.retriever

# get the query engine
query_engine = auto_merging_retriever_pack.query_engine
```
