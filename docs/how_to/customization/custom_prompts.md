# Defining Prompts

Prompting is the fundamental input that gives LLMs their expressive power. LlamaIndex uses prompts to build the index, do insertion, 
perform traversal during querying, and to synthesize the final answer.

LlamaIndex uses a set of [default prompt templates](https://github.com/jerryjliu/llama_index/blob/main/llama_index/prompts/default_prompts.py) that works well out of the box.
Users may also provide their own prompt templates to further customize the behavior of the framework.

## Defining a custom prompt

Defining a custom prompt is as simple as creating a format string

```python
from llama_index import Prompt

template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)
qa_template = Prompt(template)
```

> Note: you may see references to legacy prompt subclasses such as `QuestionAnswerPrompt`, `RefinePrompt`. These have been deprecated (and now are type aliases of `Prompt`). Now you can directly specify `Prompt(template)` to construct custom prompts. But you still have to make sure the template string contains the expected parameters (e.g. `{context_str}` and `{query_str}`) when replacing a default question answer prompt.

## Passing custom prompts into the pipeline

Since LlamaIndex is a multi-step pipeline, it's important to identify the operation that you want to modify and pass in the custom prompt at the right place.
At a high-level, prompts are used in 1) index construction, and 2) query engine execution


### Modify prompts used in index construction
Different indices use different types of prompts during construction (some don't use prompts at all). 
For instance, `GPTTreeIndex` uses a `SummaryPrompt` to hierarchically
summarize the nodes, and `GPTKeywordTableIndex` uses a `KeywordExtractPrompt` to extract keywords.

There are two equivalent ways to override the prompts:
1. via the default nodes constructor 
```python
index = GPTTreeIndex(nodes, summary_template=<custom_prompt>)
```
2. via the documents constructor.
```python
index = GPTTreeIndex.from_documents(docs, summary_template=<custom_prompt>)
```

For more details on which index uses which prompts, please visit
[Index class references](/reference/indices.rst).


### Modify prompts used in query engine
More commonly, prompts are used at query-time (i.e. for executing a query against an index and synthesizing the final response). There are also two equivalent ways to override the prompts:
1. via the high-level API
```python
query_engine = index.as_query_engine(text_qa_template=<custom_prompt>)
```
2. via the low-level composition API
```python
retriever = index.as_retriever()
synth = ResponseSynthesizer.from_args(text_qa_template=<custom_prompt>)
query_engine = RetrieverQueryEngine(retriever, response_synthesizer)
```

The two approaches above are equivalent, where 1 is essentially syntactic sugar for 2 and hides away the underlying complexity. You might want to use 1 to quickly modify some common parameters, and use 2 to have more granular control.


For more details on which classes use which prompts, please visit
[Query class references](/reference/query.rst).


## Full Example

An example can be found in [this notebook](https://github.com/jerryjliu/llama_index/blob/main/examples/paul_graham_essay/TestEssay.ipynb).


A corresponding snippet is below. We show how to define a custom prompt for question answer which
requires both a `context_str` and `query_str` field. The prompt is passed in during query-time.

```python
from llama_index import Prompt, GPTVectorStoreIndex, SimpleDirectoryReader

# load documents
documents = SimpleDirectoryReader('data').load_data()

# define custom Prompt
TEMPLATE_STR = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)
QA_TEMPLATE = Prompt(TEMPLATE_STR)

# Build index 
index = GPTVectorStoreIndex.from_documents(documents)

# Configure query engine
query_engine = index.as_query_engine(text_qa_template=QA_TEMPLATE)

# Execute query
response = query_engine.query("What did the author do growing up?")
print(response)

```


Check out the [reference documentation](/reference/prompts.rst) for a full set of all prompts.
