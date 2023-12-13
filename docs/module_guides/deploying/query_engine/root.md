# Query Engine

## Concept

Query engine is a generic interface that allows you to ask question over your data.

A query engine takes in a natural language query, and returns a rich response.
It is most often (but not always) built on one or many [indexes](/module_guides/indexing/indexing.md) via [retrievers](/module_guides/querying/retriever/root.md).
You can compose multiple query engines to achieve more advanced capability.

```{tip}
If you want to have a conversation with your data (multiple back-and-forth instead of a single question & answer), take a look at [chat Engine](/module_guides/deploying/chat_engines/root.md)
```

## Usage Pattern

Get started with:

```python
query_engine = index.as_query_engine()
response = query_engine.query("Who is Paul Graham.")
```

To stream response:

```python
query_engine = index.as_query_engine(streaming=True)
streaming_response = query_engine.query("Who is Paul Graham.")
streaming_response.print_response_stream()
```

```{toctree}
---
maxdepth: 2
---
usage_pattern.md
```

## Modules

```{toctree}
---
maxdepth: 3
---
modules.md
```

## Supporting Modules

```{toctree}
---
maxdepth: 2
---
supporting_modules.md
```
