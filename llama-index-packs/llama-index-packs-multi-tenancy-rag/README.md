# Multi-Tenancy RAG Pack

Create a Multi-Tenancy RAG using VectorStoreIndex.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack MultiTenancyRAGPack --download-dir ./multitenancy_rag_pack
```

You can then inspect the files at `./multitenancy_rag_pack` and use them as a template for your own project.

## Code Usage

You can download the pack to a the `./multitenancy_rag_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
MultiTenancyRAGPack = download_llama_pack(
    "MultiTenancyRAGPack", "./multitenancy_rag_pack"
)

# You can use any llama-hub loader to get documents and add them to index for a user!
multitenancy_rag_pack = MultiTenancyRAGPack()
multitenancy_rag_pack.add(documents, "<user>")
```

From here, you can use the pack, or inspect and modify the pack in `./multitenancy_rag_pack`.

The `run()` function is a light wrapper around `index.as_query_engine().query()`.

```python
response = multitenancy_rag_pack.run(
    "<user query>", user="<user>", similarity_top_k=2
)
```

You can also use modules individually.

```python
# Use the index directly
index = multitenancy_rag_pack.index
query_engine = index.as_query_engine(
    filters=MetadataFilters(
        filters=[
            ExactMatchFilter(
                key="user",
                value="<user>",
            )
        ]
    )
)
retriever = index.as_retriever(
    filters=MetadataFilters(
        filters=[
            ExactMatchFilter(
                key="user",
                value="<user>",
            )
        ]
    )
)
```
