# Defining Prompts

Prompting is the fundamental input that gives LLMs their expressive power. GPT Index uses prompts to build the index, do insertion, 
perform traversal during querying, and to synthesize the final answer.

GPT Index uses a finite set of *prompt types*, described [here](/reference/prompts.rst). 
All index classes, along with their associated queries, utilize a subset of these prompts. The user may provide their own prompt.
If the user does not provide their own prompt, default prompts are used.

An API reference of all index classes and query classes are found below. The definition of each index class and query
contains optional prompts that the user may pass in.
- [Indices](/reference/indices.rst)
- [Queries](/reference/query.rst)


### Example

An example can be found in [this notebook](https://github.com/jerryjliu/gpt_index/blob/main/examples/paul_graham_essay/TestEssay.ipynb).

The corresponding snippet is below. We show how to define a custom Summarization Prompt that not only
contains a `text` field, but also `query_str` field during construction of `GPTTreeIndex`, so that 
the answer to the query can be simply synthesized from the root nodes.

```python

from gpt_index import SummaryPrompt, GPTTreeIndex, SimpleDirectoryReader

# load documents
documents = SimpleDirectoryReader('data').load_data()

# define custom SummaryPrompt
query_str = "What did the author do growing up?"
SUMMARY_PROMPT_TMPL = (
    "Context information is below. \n"
    "---------------------\n"
    "{text}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    f"answer the question: {query_str}\n"
)
SUMMARY_PROMPT = SummaryPrompt(SUMMARY_PROMPT_TMPL)
# Build GPTTreeIndex: pass in custom prompt
index_with_query = GPTTreeIndex(documents, summary_template=SUMMARY_PROMPT)

```

Once the index is built, we can retrieve our answer:
```python
# directly retrieve response from root nodes instead of traversing tree
response = index_with_query.query(query_str, mode="retrieve")
```

Check out the [reference documentation](/reference/prompts.rst) for a full set of all prompts.
