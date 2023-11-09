# Prompts

## Concept

Prompting is the fundamental input that gives LLMs their expressive power. LlamaIndex uses prompts to build the index, do insertion,
perform traversal during querying, and to synthesize the final answer.

LlamaIndex uses a set of [default prompt templates](https://github.com/jerryjliu/llama_index/blob/main/llama_index/prompts/default_prompts.py) that work well out of the box.

In addition, there are some prompts written and used specifically for chat models like `gpt-3.5-turbo` [here](https://github.com/jerryjliu/llama_index/blob/main/llama_index/prompts/chat_prompts.py).

Users may also provide their own prompt templates to further customize the behavior of the framework. The best method for customizing is copying the default prompt from the link above, and using that as the base for any modifications.

## Usage Pattern

Using prompts is simple.

```python
from llama_index.prompts import PromptTemplate

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

See our Usage Pattern Guide for more details.

```{toctree}
---
maxdepth: 2
---
prompts/usage_pattern.md
```

## Example Guides

Simple Customization Examples

```{toctree}
---
maxdepth: 1
---
Completion prompts </examples/customization/prompts/completion_prompts.ipynb>
Chat prompts </examples/customization/prompts/chat_prompts.ipynb>
```

Prompt Engineering Guides

```{toctree}
---
maxdepth: 1
---
/examples/prompts/prompt_mixin.ipynb
/examples/prompts/advanced_prompts.ipynb
/examples/prompts/prompts_rag.ipynb
```

Experimental

```{toctree}
---
maxdepth: 1
---
/examples/prompts/prompt_optimization.ipynb
/examples/prompts/emotion_prompt.ipynb
```
