# Raptor Retriever LlamaPack

This LlamaPack shows how to use an implementation of RAPTOR with llama-index, leveraging the RAPTOR pack.

RAPTOR works by recursively clustering and summarizing clusters in layers for retrieval.

There two retrieval modes:

- tree_traversal -- traversing the tree of clusters, performing top-k at each level in the tree.
- collapsed -- treat the entire tree as a giant pile of nodes, perform simple top-k.

See [the paper](https://arxiv.org/abs/2401.18059) for full algorithm details.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack RaptorPack --download-dir ./raptor_pack
```

You can then inspect/modify the files at `./raptor_pack` and use them as a template for your own project.

## Code Usage

You can alternaitvely install the package:

`pip install llama-index-packs-raptor`

Then, you can import and initialize the pack! This will perform clustering and summarization over your data.

```python
from llama_index.packs.raptor import RaptorPack

pack = RaptorPack(documents, llm=llm, embed_model=embed_model)
```

The `run()` function is a light wrapper around `retriever.retrieve()`.

```python
nodes = pack.run(
    "query",
    mode="collapsed",  # or tree_traversal
)
```

You can also use modules individually.

```python
# get the retriever
retriever = pack.retriever
```

## Persistence

The `RaptorPack` comes with the `RaptorRetriever`, which offers ways of saving/reloading!

If you are using a remote vector-db, just pass it in

```python
# Pack usage
pack = RaptorPack(..., vector_store=vector_store)

# RaptorRetriever usage
retriever = RaptorRetriever(..., vector_store=vector_store)
```

Then, to re-connect, just pass in the vector store again and an empty list of documents

```python
# Pack usage
pack = RaptorPack([], ..., vector_store=vector_store)

# RaptorRetriever usage
retriever = RaptorRetriever([], ..., vector_store=vector_store)
```

Check out the [notebook here for complete details!](https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-raptor/examples/raptor.ipynb).
