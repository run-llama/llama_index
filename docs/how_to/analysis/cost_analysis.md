# Cost Analysis

Each call to an LLM will cost some amount of money - for instance, OpenAI's Davinci costs $0.02 / 1k tokens. The cost of building an index and querying depends on 

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
- `ListIndex`
- `SimpleKeywordTableIndex` - uses a regex keyword extractor to extract keywords from each document
- `GPTRAKEKeywordTableIndex` - uses a RAKE keyword extractor to extract keywords from each document

#### Indices with LLM calls
The following indices do require LLM calls during build time:
- `TreeIndex` - use LLM to hierarchically summarize the text to build the tree
- `GPTKeywordTableIndex` - use LLM to extract keywords from each document


### Query Time

There will always be >= 1 LLM call during query time, in order to synthesize the final answer. 
Some indices contain cost tradeoffs between index building and querying. `ListIndex`, for instance,
is free to build, but running a query over a list index (without filtering or embedding lookups), will
call the LLM {math}`N` times.

Here are some notes regarding each of the indices:
- `ListIndex`: by default requires {math}`N` LLM calls, where N is the number of nodes.
- `TreeIndex`: by default requires {math}`\log (N)` LLM calls, where N is the number of leaf nodes. 
    - Setting `child_branch_factor=2` will be more expensive than the default `child_branch_factor=1` (polynomial vs logarithmic), because we traverse 2 children instead of just 1 for each parent node.
- `GPTKeywordTableIndex`: by default requires an LLM call to extract query keywords.
    - Can do `index.as_retriever(retriever_mode="simple")` or `index.as_retriever(retriever_mode="rake")` to also use regex/RAKE keyword extractors on your query text.


### Token Predictor Usage

LlamaIndex offers token **predictors** to predict token usage of LLM and embedding calls.
This allows you to estimate your costs during 1) index construction, and 2) index querying, before
any respective LLM calls are made.

#### Using MockLLMPredictor

To predict token usage of LLM calls, import and instantiate the MockLLMPredictor with the following:
```python
from llama_index import MockLLMPredictor, ServiceContext

llm_predictor = MockLLMPredictor(max_tokens=256)
```

You can then use this predictor during both index construction and querying. Examples are given below.

**Index Construction**
```python
from llama_index import TreeIndex, MockLLMPredictor, SimpleDirectoryReader

documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
# the "mock" llm predictor is our token counter
llm_predictor = MockLLMPredictor(max_tokens=256)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
# pass the "mock" llm_predictor into TreeIndex during index construction
index = TreeIndex.from_documents(documents, service_context=service_context)

# get number of tokens used
print(llm_predictor.last_token_usage)
```

**Index Querying**

```python
query_engine = index.as_query_engine(
    service_context=service_context
)
response = query_engine.query("What did the author do growing up?")

# get number of tokens used
print(llm_predictor.last_token_usage)
```

#### Using MockEmbedding

You may also predict the token usage of embedding calls with `MockEmbedding`. 
You can use it in tandem with `MockLLMPredictor`.

```python
from llama_index import (
    GPTVectorStoreIndex, 
    MockLLMPredictor, 
    MockEmbedding, 
    SimpleDirectoryReader,
    ServiceContext
)

documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
index = GPTVectorStoreIndex.from_documents(documents)

# specify both a MockLLMPredictor as wel as MockEmbedding
llm_predictor = MockLLMPredictor(max_tokens=256)
embed_model = MockEmbedding(embed_dim=1536)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)

query_engine = index.as_query_engine(
    service_context=service_context
)
response = query_engine.query(
    "What did the author do after his time at Y Combinator?",
)
```


[Here is an example notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/analysis/TokenPredictor.ipynb).  


```{toctree}
---
caption: Examples
maxdepth: 1
---
../../examples/analysis/TokenPredictor.ipynb
```