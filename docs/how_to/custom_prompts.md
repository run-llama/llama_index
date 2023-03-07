# Defining Prompts

Prompting is the fundamental input that gives LLMs their expressive power. LlamaIndex uses prompts to build the index, do insertion, 
perform traversal during querying, and to synthesize the final answer.

LlamaIndex uses a finite set of *prompt types*, described [here](/reference/prompts.rst). 
All index classes, along with their associated queries, utilize a subset of these prompts. The user may provide their own prompt.
If the user does not provide their own prompt, default prompts are used.

NOTE: The majority of custom prompts are typically passed in during **query-time**, 
not during **index construction**. For instance, both the `QuestionAnswerPrompt` and `RefinePrompt` are used
during query-time to synthesize an answer. Some indices do use prompts during index construction
to build the index; for instance, `GPTTreeIndex` uses a `SummaryPrompt` to hierarchically
summarize the nodes, and `GPTKeywordTableIndex` uses a `KeywordExtractPrompt` to extract keywords.
Some indices do allow `QuestionAnswerPrompt` and `RefinePrompt` to be passed in during index
construction, but that usage is deprecated.


An API reference of all query classes and index classes (used for index construction) are found below. The definition of each query class and index class
contains optional prompts that the user may pass in.
- [Queries](/reference/query.rst)
- [Indices](/reference/indices.rst)


### Example

An example can be found in [this notebook](https://github.com/jerryjliu/gpt_index/blob/main/examples/paul_graham_essay/TestEssay.ipynb).


A corresponding snippet is below. We show how to define a custom `QuestionAnswer` prompt which
requires both a `context_str` and `query_str` field. The prompt is passed in during query-time.

```python

from llama_index import QuestionAnswerPrompt, GPTSimpleVectorIndex, SimpleDirectoryReader

# load documents
documents = SimpleDirectoryReader('data').load_data()

# define custom QuestionAnswerPrompt
query_str = "What did the author do growing up?"
QA_PROMPT_TMPL = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)
QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
# Build GPTSimpleVectorIndex
index = GPTSimpleVectorIndex(documents)

response = index.query(query_str, text_qa_template=QA_PROMPT)
print(response)

```


Check out the [reference documentation](/reference/prompts.rst) for a full set of all prompts.
