---
title: GoodMem
---

[GoodMem](https://goodmem.ai) stores documents and handles chunking, embeddings, and optional reranking on the server. Its LlamaIndex integration ingests `Document` objects and returns native `NodeWithScore` results through a `BaseRetriever`, so you can use it with query engines and agent tools.

The integration is maintained by PAIR Systems in the [goodmem-llamaindex repository](https://github.com/PAIR-Systems-Inc/goodmem-llamaindex) and published separately on [PyPI](https://pypi.org/project/llamaindex-goodmem/).

## Setup

Install the package with Python 3.10 or later:

```bash
pip install "llamaindex-goodmem>=0.2.0,<0.3"
```

You need a running GoodMem server, a space configured with an embedder, and an API key that can create and read memories in that space. A space is a collection of memories that share retrieval configuration. See the [GoodMem documentation](https://docs.goodmem.ai/) for server and space setup.

Set the REST server root, API key, and destination space:

```bash
export GOODMEM_BASE_URL="http://localhost:8080"
export GOODMEM_API_KEY="<your-api-key>"
export GOODMEM_SPACE_ID="<your-space-id>"
```

The import namespace is `llama_index.tools.goodmem`. GoodMem supplies the embeddings; you do not need to configure a LlamaIndex embedding model for these examples.

## Ingest and retrieve documents

Send whole documents to GoodMem, wait for indexing, then retrieve their chunks:

```python
import os

from goodmem import Goodmem
from llama_index.core.schema import Document
from llama_index.tools.goodmem import (
    GoodMemDocumentIngestor,
    GoodMemRetriever,
    wait_for_memories,
)

space_id = os.environ["GOODMEM_SPACE_ID"]
documents = [
    Document(
        text="Customers can return purchases within 30 days.",
        metadata={
            "source": "https://example.com/returns",
            "department": "support",
        },
    )
]

with Goodmem(
    base_url=os.environ["GOODMEM_BASE_URL"],
    api_key=os.environ["GOODMEM_API_KEY"],
) as client:
    ingestor = GoodMemDocumentIngestor(client=client, space_id=space_id)
    memory_ids = ingestor.add_documents(documents)
    wait_for_memories(client, memory_ids, timeout=120)

    retriever = GoodMemRetriever(client=client, space_ids=[space_id], top_k=3)
    nodes = retriever.retrieve("How long do customers have to return an item?")
    for result in nodes:
        print(result.score, result.text, result.metadata.get("source"))
```

`add_documents` returns accepted memory IDs immediately. Indexing is asynchronous; `wait_for_memories` checks those IDs before retrieval. If waiting times out, retain the IDs and wait again rather than uploading the documents again.

Results preserve source metadata and the document's metadata exclusions. Each node uses the GoodMem chunk ID and has a `SOURCE` relationship to its memory. Scores follow LlamaIndex's higher-is-better convention; `result.raw_score` retains the server score. Scores are not normalized to a fixed range.

## Filter retrieval

Use `MetadataFilters` to restrict results. This retriever reads connection settings from the environment:

```python
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
)

retriever = GoodMemRetriever(
    space_ids=[space_id],
    top_k=3,
    filters=MetadataFilters(
        filters=[MetadataFilter(key="department", value="support")]
    ),
)
nodes = retriever.retrieve("What is the returns policy?")
```

Supported operators are `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, and `nin`, with nested AND, OR, and single-child NOT conditions. Keys must be simple top-level names and values must be strings or finite numbers. Unsupported filters raise before a request is sent.

To rerank results, pass a configured GoodMem `reranker_id` to `GoodMemRetriever`. `top_k` controls the number of returned chunks; `fetch_k` controls the candidate count. Reranking does not require an LLM.

## Expose search as an agent tool

Wrap the filtered retriever with LlamaIndex's `RetrieverTool`. The application chooses the spaces and filters; the tool accepts a search query:

```python
from llama_index.core.tools import RetrieverTool

search_tool = RetrieverTool.from_defaults(
    retriever,
    name="search_returns_policy",
    description="Search customer support policies about returns and refunds.",
)
result = search_tool.call(
    input="How many days do I have to return a purchase?"
)
print(result.content)
```

Pass `search_tool` in the `tools` list of a LlamaIndex agent. The retriever can also be passed to `RetrieverQueryEngine.from_args` with your configured LLM. These use the same retrieved nodes as direct retrieval.

## Async retrieval

`aretrieve` uses the SDK's `AsyncGoodmem` client:

```python
import asyncio

from goodmem import AsyncGoodmem


async def search_async():
    async with AsyncGoodmem(
        base_url=os.environ["GOODMEM_BASE_URL"],
        api_key=os.environ["GOODMEM_API_KEY"],
    ) as client:
        retriever = GoodMemRetriever(
            async_client=client, space_ids=[space_id], top_k=3
        )
        return await retriever.aretrieve("What is the returns policy?")


nodes = asyncio.run(search_async())
```

In a notebook, call `await search_async()`.

For async ingestion, use `GoodMemDocumentIngestor(async_client=client, space_id=space_id).aadd_documents(...)` and `await_memories`. When injecting clients, provide `client` for sync calls and `async_client` for async calls. Caller-owned clients stay open until their enclosing context exits.

See the [usage guide](https://github.com/PAIR-Systems-Inc/goodmem-llamaindex/blob/v0.2.0/docs/usage.md) for async ingestion, retrieval diagnostics, optional administrative tools, and migration from version 0.1.
