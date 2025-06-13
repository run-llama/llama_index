# Sentence Window Retriever

This LlamaPack provides an example of our sentence window retriever.

This specific template shows the e2e process of building this. It loads
a document, chunks it up, adds surrounding context as metadata to each chunk,
and during retrieval inserts the context back into each chunk for response synthesis.

Check out the [notebook here](https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_packs/sentence_window_retriever/sentence_window.ipynb).

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack SentenceWindowRetrieverPack --download-dir ./sentence_window_retriever_pack
```

You can then inspect the files at `./sentence_window_retriever_pack` and use them as a template for your own project.

## Code Usage

You can download the pack to a the `./sentence_window_retriever_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
SentenceWindowRetrieverPack = download_llama_pack(
    "SentenceWindowRetrieverPack", "./sentence_window_retriever_pack"
)
```

From here, you can use the pack, or inspect and modify the pack in `./sentence_window_retriever_pack`.

Then, you can set up the pack like so:

```python
# create the pack
# get documents from any data loader
sentence_window_retriever_pack = SentenceWindowRetrieverPack(
    documents,
)
```

The `run()` function is a light wrapper around `query_engine.query()`.

```python
response = sentence_window_retriever_pack.run(
    "Tell me a bout a Music celebrity."
)
```

You can also use modules individually.

```python
# get the sentence vector index
index = sentence_window_retriever_pack.sentence_index

# get the node parser
node_parser = sentence_window_retriever_pack.node_parser

# get the metadata replacement postprocessor
postprocessor = sentence_window_retriever_pack.postprocessor

# get the query engine
query_engine = sentence_window_retriever_pack.query_engine
```
