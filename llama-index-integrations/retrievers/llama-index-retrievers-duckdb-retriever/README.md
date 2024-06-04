# LlamaIndex Retrievers Integration: DuckDBRetriever

## What is this?

This is a BM25 Retriever for DuckDB that can be used with LlamaIndex.

## How to use

This was created with reference to [DuckDB Full-Text Search Extension](https://duckdb.org/docs/extensions/full_text_search), so it's mostly the same.

Please refer to that.

However, while `DuckDBVectorStore` is an VectorStore, `DuckDBRetriever` is a Retriever.

DuckDBRetriever Example:

```python
retriever = DuckDBRetriever(database_name="vector.db",persist_dir="duckdb")
nodes = retriever.retrieve("retrieve_query")
```
