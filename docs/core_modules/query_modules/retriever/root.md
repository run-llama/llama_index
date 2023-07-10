
# Retriever

## Concept

Retrievers are responsible for fetching the most relevant context given a user query (or chat message).  

It can be built on top of [Indices](/core_modules/data_modules/index/root.md), but can also be defined independently.
It is used as a key building block in [Query Engines](/core_modules/query_modules/query_engine/root.md) (and [Chat Engines](/core_modules/query_modules/chat_engines/root.md)) for retrieving relevant context.

```{tip}
Confused about where retriever fits in the pipeline? Read about [high-level concepts](/getting_started/concepts.md)
```

## Usage Pattern

Get started with:
```python
retriever = index.as_retriever()
nodes = retriever.retrieve("Who is Paul Graham?")
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
maxdepth: 2
---
modules.md
```