# Prompt Usage Pattern

## Using `RichPromptTemplate` and Jinja Syntax

By leveraging Jinja syntax, you can build prompt templates that have variables, logic, parse objects, and more.

Let's look at a few examples:

```python
from llama_index.core.prompts import RichPromptTemplate

template = RichPromptTemplate(
    """We have provided context information below.
---------------------
{{ context_str }}
---------------------
Given this information, please answer the question: {{ query_str }}
"""
)

# format as a string
prompt_str = template.format(context_str=..., query_str=...)

# format as a list of chat messages
messages = template.format_messages(context_str=..., query_str=...)
```

The main difference between Jinja prompts and f-strings is the variables now have double brackets `{{ }}` instead of single brackets `{ }`.

Let look at a more complex example that uses loops to generate a multi-modal prompt.

```python
from llama_index.core.prompts import RichPromptTemplate

template = RichPromptTemplate(
    """
{% chat role="system" %}
Given a list if images and text from each image, please answer the question to the best of your ability.
{% endchat %}

{% chat role="user" %}
{% for image_path, text in images_and_texts %}
Here is some text: {{ text }}
And here is an image:
{{ image_path | image }}
{% endfor %}
{% endchat %}
"""
)

messages = template.format_messages(
    images_and_texts=[
        ("page_1.png", "This is the first page of the document"),
        ("page_2.png", "This is the second page of the document"),
    ]
)
```

In this example, you can see several features:

- the `{% chat %}` block is used to format the message as a chat message and set the role
- the `{% for %}` loop is used to iterate over the `images_and_texts` list that is passed in
- the `{{ image_path | image }}` syntax is used to format the image path as an image content block. Here, `|` is used to apply a "filter" to the variable to help identify it as an image.


Let's look at another example, this time for creating a template using nodes from a retriever:

```python
from llama_index.core.prompts import RichPromptTemplate

template = RichPromptTemplate(
    """
{% chat role="system" %}
You are a helpful assistant that can answer questions about the context provided.
{% endchat %}

{% chat role="user" %}
{% for node in nodes %}
{{ node.text }}
{% endfor %}
{% endchat %}
"""
)

nodes = retriever.retrieve("What is the capital of the moon?")

messages = template.format_messages(nodes=nodes)
```

## Using `f-string` Prompt Templates

As of this writing, many older components and examples will use `f-string` prompts.`

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

## Getting and Setting Custom Prompts

Since LlamaIndex is a multi-step pipeline, it's important to identify the operation that you want to modify and pass in the custom prompt at the right place.

For instance, prompts are used in response synthesizer, retrievers, index construction, etc; some of these modules are nested in other modules (synthesizer is nested in query engine).

See [this guide](../../../examples/prompts/prompt_mixin.ipynb) for full details on accessing/customizing prompts.

### Commonly Used Prompts

The most commonly used prompts will be the `text_qa_template` and the `refine_template`.

- `text_qa_template` - used to get an initial answer to a query using retrieved nodes
- `refine_template` - used when the retrieved text does not fit into a single LLM call with `response_mode="compact"` (the default), or when more than one node is retrieved using `response_mode="refine"`. The answer from the first query is inserted as an `existing_answer`, and the LLM must update or repeat the existing answer based on the new context.

### Accessing Prompts

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

### Updating Prompts

You can customize prompts on any module that implements `get_prompts` with the `update_prompts` function. Just pass in argument values with the keys equal to the keys you see in the prompt dictionary
obtained through `get_prompts`.

e.g. regarding the example above, we might do the following

```python
# shakespeare!
qa_prompt_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{{ context_str }}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query in the style of a Shakespeare play.\n"
    "Query: {{ query_str }}\n"
    "Answer: "
)
qa_prompt_tmpl = RichPromptTemplate(qa_prompt_tmpl_str)

query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
)
```

### Modify prompts used in query engine

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

Check out the [reference documentation](../../../api_reference/prompts/index.md) for a full set of all prompts and their methods/parameters.

## [Advanced] Advanced Prompt Capabilities

In this section we show some advanced prompt capabilities in LlamaIndex.

Related Guides:

- [Advanced Prompts](../../../examples/prompts/advanced_prompts.ipynb)
- [RichPromptTemplate Features](../../../examples/prompts/rich_prompt_template_features.ipynb)

### Function Mappings

Pass in functions as template variables instead of fixed values.

This is quite advanced and powerful; allows you to do dynamic few-shot prompting, etc.

Here's an example of reformatting the `context_str`.

```python
from llama_index.core.prompts import RichPromptTemplate


def format_context_fn(**kwargs):
    # format context with bullet points
    context_list = kwargs["context_str"].split("\n\n")
    fmtted_context = "\n\n".join([f"- {c}" for c in context_list])
    return fmtted_context


prompt_tmpl = RichPromptTemplate(
    "{{ context_str }}", function_mappings={"context_str": format_context_fn}
)

prompt_str = prompt_tmpl.format(context_str="context", query_str="query")
```

### Partial Formatting

Partially format a prompt, filling in some variables while leaving others to be filled in later.

```python
from llama_index.core.prompts import RichPromptTemplate

template = RichPromptTemplate(
    """
{{ foo }} {{ bar }}
"""
)

partial_prompt_tmpl = template.partial_format(foo="abc")

fmt_str = partial_prompt_tmpl.format(bar="def")
```

### Template Variable Mappings

LlamaIndex prompt abstractions generally expect certain keys. E.g. our `text_qa_prompt` expects `context_str` for context and `query_str` for the user query.

But if you're trying to adapt a string template for use with LlamaIndex, it can be annoying to change out the template variables.

Instead, define `template_var_mappings`:

```python
from llama_index.core.prompts import RichPromptTemplate

template_var_mappings = {"context_str": "my_context", "query_str": "my_query"}

prompt_tmpl = RichPromptTemplate(
    "Here is some context: {{ context_str }} and here is a query: {{ query_str }}",
    template_var_mappings=template_var_mappings,
)

prompt_str = prompt_tmpl.format(my_context="context", my_query="query")
```
