# Cost Analysis

Each call to an LLM will cost some amount of money - for instance, OpenAI's Davinci costs $0.02 / 1k tokens. The cost of building an index and querying depends on 

- the type of LLM used
- the type of data structure used
- parameters used during building 
- parameters used during querying

The cost of building and querying each index is a TODO in the reference documentation. In the meantime, we provide the following information:

1. A high-level overview of the cost structure of the indices.
2. A token predictor that you can use directly within GPT Index!

### Overview of Cost Structure

#### Indices with no LLM calls
The following indices don't require LLM calls at all during building (0 cost):
- `GPTListIndex`
- `GPTSimpleKeywordTableIndex` - uses a regex keyword extractor to extract keywords from each document
- `GPTRAKEKeywordTableIndex` - uses a RAKE keyword extractor to extract keywords from each document

#### Indices with LLM calls
The following indices do require LLM calls during build time:
- `GPTTreeIndex` - use LLM to hierarchically summarize the text to build the tree
- `GPTKeywordTableIndex` - use LLM to extract keywords from each document


### Query Time

There will always be >= 1 LLM call during query time, in order to synthesize the final answer. 
Some indices contain cost tradeoffs between index building and querying. `GPTListIndex`, for instance,
is free to build, but running a query over a list index (without filtering or embedding lookups), will
call the LLM {math}`N` times.

Here are some notes regarding each of the indices:
- `GPTListIndex`: by default requires {math}`N` LLM calls, where N is the number of nodes.
    - However, can do `index.query(..., keyword="<keyword>")` to filter out nodes that don't contain the keyword
- `GPTTreeIndex`: by default requires {math}`\log (N)` LLM calls, where N is the number of leaf nodes. 
    - Setting `child_branch_factor=2` will be more expensive than the default `child_branch_factor=1` (polynomial vs logarithmic), because we traverse 2 children instead of just 1 for each parent node.
- `GPTKeywordTableIndex`: by default requires an LLM call to extract query keywords.
    - Can do `index.query(..., mode="simple")` or `index.query(..., mode="rake")` to also use regex/RAKE keyword extractors on your query text.


### Token Predictor Usage

GPT Index offers a token predictor, allowing you to estimate your costs during 1) index construction, and 2) index querying, before
any respective LLM calls are made.

To do this, import and instantiate the MockLLMPredictor with the following:
```python
from gpt_index import MockLLMPredictor

llm_predictor = MockLLMPredictor(max_tokens=256)
```

You can then use this predictor during both index construction and querying. Examples are given below.

**Index Construction**

![](/_static/cost_analysis/build.png)

**Index Querying**

![](/_static/cost_analysis/query.png)


[Here is an example notebook](https://github.com/jerryjliu/gpt_index/blob/main/examples/cost_analysis/TokenPredictor.ipynb).