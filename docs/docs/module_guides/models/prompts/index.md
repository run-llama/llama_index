# Prompts

## Concept

Prompting is the fundamental input that gives LLMs their expressive power. LlamaIndex uses prompts to build the index, do insertion, perform traversal during querying, and to synthesize the final answer.

When building agentic workflows, building and managing prompts is a key part of the development process. LlamaIndex provides a flexible and powerful way to manage prompts, and to use them in a variety of ways.

- `RichPromptTemplate` - latest-style for building jinja-style prompts with variables and logic
- `PromptTemplate` - older-style simple templating for building prompts with a single f-string
- `ChatPromptTemplate` - older-style simple templating for building chat prompts with messages and f-strings

LlamaIndex uses a set of [default prompt templates](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/prompts/default_prompts.py) that work well out of the box.

In addition, there are some prompts written and used specifically for chat models like `gpt-3.5-turbo` [here](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/prompts/chat_prompts.py).

Users may also provide their own prompt templates to further customize the behavior of the framework. The best method for customizing is copying the default prompt from the link above, and using that as the base for any modifications.

## Usage Pattern

Using prompts is simple. Below is an example of how to use the `RichPromptTemplate` to build a jinja-style prompt template:

```python
from llama_index.core.prompts import RichPromptTemplate

template_str = """We have provided context information below.
---------------------
{{ context_str }}
---------------------
Given this information, please answer the question: {{ query_str }}
"""
qa_template = RichPromptTemplate(template_str)

# you can create text prompt (for completion API)
prompt = qa_template.format(context_str=..., query_str=...)

# or easily convert to message prompts (for chat API)
messages = qa_template.format_messages(context_str=..., query_str=...)
```

See our [Usage Pattern Guide](./usage_pattern.md) for more details on taking full advantage of the `RichPromptTemplate` and details on the other prompt templates.

## Example Guides

Prompt Engineering Guides

- [Advanced Prompts](../../../examples/prompts/advanced_prompts.ipynb)
- [RichPromptTemplate Features](../../../examples/prompts/rich_prompt_template_features.ipynb)

Simple Customization Examples

- [Completion prompts](../../../examples/customization/prompts/completion_prompts.ipynb)
- [Chat prompts](../../../examples/customization/prompts/chat_prompts.ipynb)
- [Prompt Mixin](../../../examples/prompts/prompt_mixin.ipynb)

Experimental

- [Emotion Prompting](../../../examples/prompts/emotion_prompt.ipynb)
