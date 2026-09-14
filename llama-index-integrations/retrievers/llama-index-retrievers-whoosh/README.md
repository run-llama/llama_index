# llama-index-retrievers-whoosh

A pure-Python **BM25 (lexical) retriever for [LlamaIndex](https://github.com/run-llama/llama_index)**, powered by [Whoosh](https://github.com/priya-sundaram-dev/whoosh).

LlamaIndex pipelines usually reach for a *vector* index, but dense retrieval has a
well-known blind spot: it can quietly miss the *exact* tokens that matter most —
product SKUs, function names, error codes like `ERR_2043`, gene symbols, ticket
IDs. A lexical BM25 retriever is the classic complement, and Whoosh gives you one
in **pure Python**: no server, no native wheels, and an index that is just a
folder on disk.

## Install

```bash
pip install llama-index-retrievers-whoosh
```

This pulls in `llama-index-core` and `whoosh3` (the maintained Whoosh fork).

## Quick start

```python
from llama_index.retrievers.whoosh import WhooshRetriever

retriever = WhooshRetriever.from_texts(
    texts=[
        "Whoosh is a pure-Python full-text search library.",
        "BM25 ranks documents by term rarity and frequency.",
    ],
    ids=["a", "b"],
    metadatas=[{"src": "readme"}, {"src": "docs"}],
    k=4,
)

nodes = retriever.retrieve("pure python search")
for n in nodes:
    print(n.score, n.node.metadata["id"], n.node.text)
```

Every result is a standard LlamaIndex `NodeWithScore`, so it drops straight into
any query engine or router.

## Persist to disk

Pass a `path` to build an on-disk index once, then reopen it later:

```python
WhooshRetriever.from_texts(texts=texts, ids=ids, path="./whoosh_index")
retriever = WhooshRetriever.from_index("./whoosh_index", k=8)
```

The index is just a directory — copy it, commit it, ship it in a container.

## Hybrid (lexical + vector) search

Combine this retriever with any vector retriever using LlamaIndex's
`QueryFusionRetriever`, which does Reciprocal Rank Fusion for you:

```python
from llama_index.core.retrievers import QueryFusionRetriever

fusion = QueryFusionRetriever(
    [whoosh_retriever, vector_retriever],
    similarity_top_k=8,
    num_queries=1,          # set >1 to also fuse query rewrites
    mode="reciprocal_rerank",
)
nodes = fusion.retrieve("ERR_2043 timeout after upgrade")
```

Lexical retrieval catches the exact `ERR_2043` token; the vector retriever
catches the paraphrases. Fusing them is consistently stronger than either alone.

## Why Whoosh?

- **Pure Python** — no Java, no C extensions, no server to run.
- **BM25F ranking** out of the box.
- **The index is a folder** — trivial to build in CI, cache, or ship.

## License

BSD-2-Clause, matching Whoosh. See the [Whoosh repository](https://github.com/priya-sundaram-dev/whoosh) for the underlying library and its history.
