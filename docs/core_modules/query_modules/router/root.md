# Routers

## Concept
Routers are modules that take in a user query and a set of "choices" (defined by metadata), and returns one or more selected choices.

They can be used on their own (as "selector modules"), or used as a query engine or retriever (e.g. on top of other query engines/retrievers).

They are simple but powerful modules that use LLMs for decision making capabilities. They can be used for the following use cases and more:
- Selecting the right data source among a diverse range of data sources
- Deciding whether to do summarization (e.g. using summary index query engine) or semantic search (e.g. using vector index query engine)
- Deciding whether to "try" out a bunch of choices at once and combine the results (using multi-routing capabilities).

The core router modules exist in the following forms:
- LLM selectors put the choices as a text dump into a prompt and use LLM text completion endpoint to make decisions
- Pydantic selectors pass choices as Pydantic schemas into a function calling endpoint, and return Pydantic objects

## Usage Pattern

A simple example of using our router module as part of a query engine is given below.

```python
from llama_index.query_engine.router_query_engine import RouterQueryEngine
from llama_index.selectors.pydantic_selectors import PydanticSingleSelector
from llama_index.tools.query_engine import QueryEngineTool


list_tool = QueryEngineTool.from_defaults(
    query_engine=list_query_engine,
    description="Useful for summarization questions related to the data source",
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Useful for retrieving specific context related to the data source",
)

query_engine = RouterQueryEngine(
    selector=PydanticSingleSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
    ],
)
query_engine.query("<query>")
```

You can find more details using routers as standalone modules, as part of a query engine, and as part of a retriever
below in the usage pattern guide.

```{toctree}
---
maxdepth: 2
---
usage_pattern.md
```

## Modules
Below you can find extensive guides using routers in different settings.

```{toctree}
---
maxdepth: 2
---
modules.md
```