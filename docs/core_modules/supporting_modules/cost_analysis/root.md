# Cost Analysis

## Concept
Each call to an LLM will cost some amount of money - for instance, OpenAI's gpt-3.5-turbo costs $0.002 / 1k tokens. The cost of building an index and querying depends on 

- the type of LLM used
- the type of data structure used
- parameters used during building 
- parameters used during querying

The cost of building and querying each index is a TODO in the reference documentation. In the meantime, we provide the following information:

1. A high-level overview of the cost structure of the indices.
2. A token predictor that you can use directly within LlamaIndex!

### Overview of Cost Structure

#### Indices with no LLM calls
The following indices don't require LLM calls at all during building (0 cost):
- `SummaryIndex`
- `SimpleKeywordTableIndex` - uses a regex keyword extractor to extract keywords from each document
- `RAKEKeywordTableIndex` - uses a RAKE keyword extractor to extract keywords from each document

#### Indices with LLM calls
The following indices do require LLM calls during build time:
- `TreeIndex` - use LLM to hierarchically summarize the text to build the tree
- `KeywordTableIndex` - use LLM to extract keywords from each document

### Query Time

There will always be >= 1 LLM call during query time, in order to synthesize the final answer. 
Some indices contain cost tradeoffs between index building and querying. `SummaryIndex`, for instance,
is free to build, but running a query over a summary index (without filtering or embedding lookups), will
call the LLM {math}`N` times.

Here are some notes regarding each of the indices:
- `SummaryIndex`: by default requires {math}`N` LLM calls, where N is the number of nodes.
- `TreeIndex`: by default requires {math}`\log (N)` LLM calls, where N is the number of leaf nodes. 
    - Setting `child_branch_factor=2` will be more expensive than the default `child_branch_factor=1` (polynomial vs logarithmic), because we traverse 2 children instead of just 1 for each parent node.
- `KeywordTableIndex`: by default requires an LLM call to extract query keywords.
    - Can do `index.as_retriever(retriever_mode="simple")` or `index.as_retriever(retriever_mode="rake")` to also use regex/RAKE keyword extractors on your query text.
-  `VectorStoreIndex`: by default, requires one LLM call per query. If you increase the `similarity_top_k` or `chunk_size`, or change the `response_mode`, then this number will increase.

## Usage Pattern

LlamaIndex offers token **predictors** to predict token usage of LLM and embedding calls.
This allows you to estimate your costs during 1) index construction, and 2) index querying, before
any respective LLM calls are made.

Tokens are counted using the `TokenCountingHandler` callback. See the [example notebook](../../../examples/callbacks/TokenCountingHandler.ipynb) for details on the setup.

### Using MockLLM

To predict token usage of LLM calls, import and instantiate the MockLLM as shown below. The `max_tokens` parameter is used as a "worst case" prediction, where each LLM response will contain exactly that number of tokens. If `max_tokens` is not specified, then it will simply predict back the prompt.

```python
from llama_index import ServiceContext, set_global_service_context
from llama_index.llms import MockLLM

llm = MockLLM(max_tokens=256)

service_context = ServiceContext.from_defaults(llm=llm)

# optionally set a global service context
set_global_service_context(service_context)
```

You can then use this predictor during both index construction and querying. 

### Using MockEmbedding

You may also predict the token usage of embedding calls with `MockEmbedding`. 

```python
from llama_index import ServiceContext, set_global_service_context
from llama_index import MockEmbedding

# specify a MockLLMPredictor
embed_model = MockEmbedding(embed_dim=1536)

service_context = ServiceContext.from_defaults(embed_model=embed_model)

# optionally set a global service context
set_global_service_context(service_context)
```

## Usage Pattern

Read about the full usage pattern below!

```{toctree}
---
caption: Examples
maxdepth: 1
---
usage_pattern.md
```
