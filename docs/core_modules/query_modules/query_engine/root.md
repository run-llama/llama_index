# Query Engine

## Concept
Query engine is a generic interface that allows you to ask question over your data.
> If you want to have a conversation with your data (multiple back-and-forth instead of a single question & answer), take a look at [Chat Engine](/how_to/chat_engine/root.md)  

A query engine takes in a natural language query, and returns a rich response.
It is most often (but not always) built on one or many [Indices](/how_to/index/root.md) via [Retrievers](/how_to/retriever/root.md).
You can compose multiple query engines to achieve more advanced capability.


## Usage Pattern
Get started with:
```python
query_engine = index.as_query_engine()
response = query_engine.query("Who is Paul Graham.")
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
