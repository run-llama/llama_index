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
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import QueryEngineTool


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

## Usage Pattern

Defining a "selector" is at the core of defining a router.

You can easily use our routers as a query engine or a retriever. In these cases, the router will be responsible
for "selecting" query engine(s) or retriever(s) to route the user query to.

We also highlight our `ToolRetrieverRouterQueryEngine` for retrieval-augmented routing - this is the case
where the set of choices themselves may be very big and may need to be indexed. **NOTE**: this is a beta feature.

We also highlight using our router as a standalone module.

## Defining a selector

Some examples are given below with LLM and Pydantic based single/multi selectors:

```python
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)

# pydantic selectors feed in pydantic objects to a function calling API
# single selector (pydantic)
selector = PydanticSingleSelector.from_defaults()
# multi selector (pydantic)
selector = PydanticMultiSelector.from_defaults()

# LLM selectors use text completion endpoints
# single selector (LLM)
selector = LLMSingleSelector.from_defaults()
# multi selector (LLM)
selector = LLMMultiSelector.from_defaults()
```

## Using as a Query Engine

A `RouterQueryEngine` is composed on top of other query engines as tools.

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.selectors.pydantic_selectors import Pydantic
from llama_index.core.tools import QueryEngineTool
from llama_index.core import VectorStoreIndex, SummaryIndex

# define query engines
...

# initialize tools
list_tool = QueryEngineTool.from_defaults(
    query_engine=list_query_engine,
    description="Useful for summarization questions related to the data source",
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Useful for retrieving specific context related to the data source",
)

# initialize router query engine (single selection, pydantic)
query_engine = RouterQueryEngine(
    selector=PydanticSingleSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
    ],
)
query_engine.query("<query>")
```

## Using as a Retriever

Similarly, a `RouterRetriever` is composed on top of other retrievers as tools. An example is given below:

```python
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import RetrieverTool

# define indices
...

# define retrievers
vector_retriever = vector_index.as_retriever()
keyword_retriever = keyword_index.as_retriever()

# initialize tools
vector_tool = RetrieverTool.from_defaults(
    retriever=vector_retriever,
    description="Useful for retrieving specific context from Paul Graham essay on What I Worked On.",
)
keyword_tool = RetrieverTool.from_defaults(
    retriever=keyword_retriever,
    description="Useful for retrieving specific context from Paul Graham essay on What I Worked On (using entities mentioned in query)",
)

# define retriever
retriever = RouterRetriever(
    selector=PydanticSingleSelector.from_defaults(llm=llm),
    retriever_tools=[
        list_tool,
        vector_tool,
    ],
)
```

## Using selector as a standalone module

You can use the selectors as standalone modules. Define choices as either a list of `ToolMetadata` or as a list of strings.

```python
from llama_index.core.tools import ToolMetadata
from llama_index.core.selectors import LLMSingleSelector


# choices as a list of tool metadata
choices = [
    ToolMetadata(description="description for choice 1", name="choice_1"),
    ToolMetadata(description="description for choice 2", name="choice_2"),
]

# choices as a list of strings
choices = [
    "choice 1 - description for choice 1",
    "choice 2: description for choice 2",
]

selector = LLMSingleSelector.from_defaults()
selector_result = selector.select(
    choices, query="What's revenue growth for IBM in 2007?"
)
print(selector_result.selections)
```

More examples:

- [Router Query Engine](../../../examples/query_engine/RouterQueryEngine.ipynb)
- [Retriever Router Query Engine](../../../examples/query_engine/RetrieverRouterQueryEngine.ipynb)
- [SQL Router Query Engine](../../../examples/query_engine/SQLRouterQueryEngine.ipynb)
- [Router Retriever](../../../examples/retrievers/router_retriever.ipynb)
