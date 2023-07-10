# Prompts

## Concept

Prompting is the fundamental input that gives LLMs their expressive power. LlamaIndex uses prompts to build the index, do insertion, 
perform traversal during querying, and to synthesize the final answer.

LlamaIndex uses a set of [default prompt templates](https://github.com/jerryjliu/llama_index/blob/main/llama_index/prompts/default_prompts.py) that work well out of the box.

In addition, there are some prompts written and used specifically for chat models like `gpt-3.5-turbo` [here](https://github.com/jerryjliu/llama_index/blob/main/llama_index/prompts/chat_prompts.py).

Users may also provide their own prompt templates to further customize the behavior of the framework. The best method for customizing is copying the default prompt from the link above, and using that as the base for any modifications.

## Usage Pattern

### Defining a custom prompt

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

### Passing custom prompts into the pipeline

Since LlamaIndex is a multi-step pipeline, it's important to identify the operation that you want to modify and pass in the custom prompt at the right place.

At a high-level, prompts are used in 1) index construction, and 2) query engine execution

The most commonly used prompts will be the `text_qa_template` and the `refine_template`. 

- `text_qa_template` - used to get an initial answer to a query using retrieved nodes
- `refine_tempalate` - used when the retrieved text does not fit into a single LLM call with `response_mode="compact"` (the default), or when more than one node is retrieved using `response_mode="refine"`. The answer from the first query is inserted as an `existing_answer`, and the LLM must update or repeat the existing answer based on the new context.

#### Modify prompts used in index construction
Different indices use different types of prompts during construction (some don't use prompts at all). 
For instance, `TreeIndex` uses a `SummaryPrompt` to hierarchically
summarize the nodes, and `KeywordTableIndex` uses a `KeywordExtractPrompt` to extract keywords.

There are two equivalent ways to override the prompts:

1. via the default nodes constructor 

```python
index = TreeIndex(nodes, summary_template=<custom_prompt>)
```
2. via the documents constructor.

```python
index = TreeIndex.from_documents(docs, summary_template=<custom_prompt>)
```

For more details on which index uses which prompts, please visit
[Index class references](/api_reference/indices.rst).

#### Modify prompts used in query engine
More commonly, prompts are used at query-time (i.e. for executing a query against an index and synthesizing the final response). 

There are also two equivalent ways to override the prompts:

1. via the high-level API
```python
query_engine = index.as_query_engine(
    text_qa_template=<custom_qa_prompt>,
    refine_template=<custom_refine_prompt>
)
```
2. via the low-level composition API

```python
retriever = index.as_retriever()
synth = get_response_synthesizer(
    text_qa_template=<custom_qa_prompt>,
    refine_template=<custom_refine_prompt>
)
query_engine = RetrieverQueryEngine(retriever, response_synthesizer)
```

The two approaches above are equivalent, where 1 is essentially syntactic sugar for 2 and hides away the underlying complexity. You might want to use 1 to quickly modify some common parameters, and use 2 to have more granular control.


For more details on which classes use which prompts, please visit
[Query class references](/api_reference/query.rst).

Check out the [reference documentation](/api_reference/prompts.rst) for a full set of all prompts.

## Modules

```{toctree}
---
maxdepth: 1
---
/examples/customization/prompts/completion_prompts.ipynb
/examples/customization/prompts/chat_prompts.ipynb
```