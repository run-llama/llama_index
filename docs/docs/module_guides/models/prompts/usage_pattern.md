## Usage Pattern

### Defining a custom prompt

Defining a custom prompt is as simple as creating a format string

```python
from llama_index.core import PromptTemplate

template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)
qa_template = PromptTemplate(template)

# you can create text prompt (for completion API)
prompt = qa_template.format(context_str=..., query_str=...)

# or easily convert to message prompts (for chat API)
messages = qa_template.format_messages(context_str=..., query_str=...)
```

> Note: you may see references to legacy prompt subclasses such as `QuestionAnswerPrompt`, `RefinePrompt`. These have been deprecated (and now are type aliases of `PromptTemplate`). Now you can directly specify `PromptTemplate(template)` to construct custom prompts. But you still have to make sure the template string contains the expected parameters (e.g. `{context_str}` and `{query_str}`) when replacing a default question answer prompt.

You can also define a template from chat messages

```python
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole

message_templates = [
    ChatMessage(content="You are an expert system.", role=MessageRole.SYSTEM),
    ChatMessage(
        content="Generate a short story about {topic}",
        role=MessageRole.USER,
    ),
]
chat_template = ChatPromptTemplate(message_templates=message_templates)

# you can create message prompts (for chat API)
messages = chat_template.format_messages(topic=...)

# or easily convert to text prompt (for completion API)
prompt = chat_template.format(topic=...)
```

### Getting and Setting Custom Prompts

Since LlamaIndex is a multi-step pipeline, it's important to identify the operation that you want to modify and pass in the custom prompt at the right place.

For instance, prompts are used in response synthesizer, retrievers, index construction, etc; some of these modules are nested in other modules (synthesizer is nested in query engine).

See [this guide](../../../examples/prompts/prompt_mixin.ipynb) for full details on accessing/customizing prompts.

#### Commonly Used Prompts

The most commonly used prompts will be the `text_qa_template` and the `refine_template`.

- `text_qa_template` - used to get an initial answer to a query using retrieved nodes
- `refine_template` - used when the retrieved text does not fit into a single LLM call with `response_mode="compact"` (the default), or when more than one node is retrieved using `response_mode="refine"`. The answer from the first query is inserted as an `existing_answer`, and the LLM must update or repeat the existing answer based on the new context.

#### Accessing Prompts

You can call `get_prompts` on many modules in LlamaIndex to get a flat list of prompts used within the module and nested submodules.

For instance, take a look at the following snippet.

```python
query_engine = index.as_query_engine(response_mode="compact")
prompts_dict = query_engine.get_prompts()
print(list(prompts_dict.keys()))
```

You might get back the following keys:

```
['response_synthesizer:text_qa_template', 'response_synthesizer:refine_template']
```

Note that prompts are prefixed by their sub-modules as "namespaces".

#### Updating Prompts

You can customize prompts on any module that implements `get_prompts` with the `update_prompts` function. Just pass in argument values with the keys equal to the keys you see in the prompt dictionary
obtained through `get_prompts`.

e.g. regarding the example above, we might do the following

```python
# shakespeare!
qa_prompt_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query in the style of a Shakespeare play.\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
)
```

#### Modify prompts used in query engine

For query engines, you can also pass in custom prompts directly during query-time (i.e. for executing a query against an index and synthesizing the final response).

There are also two equivalent ways to override the prompts:

1. via the high-level API

```python
query_engine = index.as_query_engine(
    text_qa_template=custom_qa_prompt, refine_template=custom_refine_prompt
)
```

2. via the low-level composition API

```python
retriever = index.as_retriever()
synth = get_response_synthesizer(
    text_qa_template=custom_qa_prompt, refine_template=custom_refine_prompt
)
query_engine = RetrieverQueryEngine(retriever, response_synthesizer)
```

The two approaches above are equivalent, where 1 is essentially syntactic sugar for 2 and hides away the underlying complexity. You might want to use 1 to quickly modify some common parameters, and use 2 to have more granular control.

For more details on which classes use which prompts, please visit
[Query class references](../../../api_reference/response_synthesizers/index.md).

Check out the [reference documentation](../../../api_reference/prompts/index.md) for a full set of all prompts.

#### Modify prompts used in index construction

Some indices use different types of prompts during construction
(**NOTE**: the most common ones, `VectorStoreIndex` and `SummaryIndex`, don't use any).

For instance, `TreeIndex` uses a summary prompt to hierarchically
summarize the nodes, and `KeywordTableIndex` uses a keyword extract prompt to extract keywords.

There are two equivalent ways to override the prompts:

1. via the default nodes constructor

```python
index = TreeIndex(nodes, summary_template=custom_prompt)
```

2. via the documents constructor.

```python
index = TreeIndex.from_documents(docs, summary_template=custom_prompt)
```

For more details on which index uses which prompts, please visit
[Index class references](../../../api_reference/indices/index.md).

### [Advanced] Advanced Prompt Capabilities

In this section we show some advanced prompt capabilities in LlamaIndex.

Related Guides:

- [Advanced Prompts](../../../examples/prompts/advanced_prompts.ipynb)
- [Prompt Engineering for RAG](../../../examples/prompts/prompts_rag.ipynb)

#### Partial Formatting

Partially format a prompt, filling in some variables while leaving others to be filled in later.

```python
from llama_index.core import PromptTemplate

prompt_tmpl_str = "{foo} {bar}"
prompt_tmpl = PromptTemplate(prompt_tmpl_str)
partial_prompt_tmpl = prompt_tmpl.partial_format(foo="abc")

fmt_str = partial_prompt_tmpl.format(bar="def")
```

#### Template Variable Mappings

LlamaIndex prompt abstractions generally expect certain keys. E.g. our `text_qa_prompt` expects `context_str` for context and `query_str` for the user query.

But if you're trying to adapt a string template for use with LlamaIndex, it can be annoying to change out the template variables.

Instead, define `template_var_mappings`:

```python
template_var_mappings = {"context_str": "my_context", "query_str": "my_query"}

prompt_tmpl = PromptTemplate(
    qa_prompt_tmpl_str, template_var_mappings=template_var_mappings
)
```

#### Function Mappings

Pass in functions as template variables instead of fixed values.

This is quite advanced and powerful; allows you to do dynamic few-shot prompting, etc.

Here's an example of reformatting the `context_str`.

```python
def format_context_fn(**kwargs):
    # format context with bullet points
    context_list = kwargs["context_str"].split("\n\n")
    fmtted_context = "\n\n".join([f"- {c}" for c in context_list])
    return fmtted_context


prompt_tmpl = PromptTemplate(
    qa_prompt_tmpl_str, function_mappings={"context_str": format_context_fn}
)

prompt_tmpl.format(context_str="context", query_str="query")
```
