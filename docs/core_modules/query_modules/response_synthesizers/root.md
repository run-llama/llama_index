# 📨 Response Synthesizers

## Concept
A `Response Synthesizer` is what generates a response from an LLM, using a user query and a given set of text chunks. The output of a response synthesizer is a `Response` object.

The method for doing this can take many forms, from as simple as iterating over text chunks, to as complex as building a tree. The main idea here is to simplify the process of generating a response using an LLM across your data.

When used in a query engine, the response synthesizer is used after nodes are retrieved from a retriever, and after any node-postprocessors are ran.

## Usage Pattern
Use a response synthesizer on it's own:

```python
from llama_index.schema import Node
from llama_index.response_synthesizers import get_response_synthesizer

response_synthesizer = get_response_synthesizer(response_mode='compact')

response = response_synthesizer.synthesize("query text", nodes=[Node(text="text"), ...])
```

Or in a query engine after you've created an index:

```python
query_engine = index.as_query_engine(response_synthesizer=response_synthesizer)
response = query_engine.query("query_text")
```

You can find more details on all available response synthesizers, modes, and how to build your own below.

```{toctree}
---
maxdepth: 2
---
usage_pattern.md
```

## Modules
Below you can find detailed API information for each response synthesis module.

```{toctree}
---
maxdepth: 1
---
modules.md
```