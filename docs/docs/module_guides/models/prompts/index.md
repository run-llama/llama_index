# Prompts

## Concept

Prompting is the fundamental input that gives LLMs their expressive power. LlamaIndex uses prompts to build the index, do insertion,
perform traversal during querying, and to synthesize the final answer.

LlamaIndex uses a set of [default prompt templates](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/prompts/default_prompts.py) that work well out of the box.

In addition, there are some prompts written and used specifically for chat models like `gpt-3.5-turbo` [here](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/prompts/chat_prompts.py).

Users may also provide their own prompt templates to further customize the behavior of the framework. The best method for customizing is copying the default prompt from the link above, and using that as the base for any modifications.

## Usage Pattern

Using prompts is simple.

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

See our [Usage Pattern Guide](./usage_pattern.md) for more details.

## Example Guides

Simple Customization Examples

- [Completion prompts](../../../examples/customization/prompts/completion_prompts.ipynb)
- [Chat prompts](../../../examples/customization/prompts/chat_prompts.ipynb)
- [Prompt Mixin](../../../examples/prompts/prompt_mixin.ipynb)

Prompt Engineering Guides

- [Advanced Prompts](../../../examples/prompts/advanced_prompts.ipynb)
- [RAG Prompts](../../../examples/prompts/prompts_rag.ipynb)

Experimental

- [Prompt Optimization](../../../examples/prompts/prompt_optimization.ipynb)
- [Emotion Prompting](../../../examples/prompts/emotion_prompt.ipynb)
