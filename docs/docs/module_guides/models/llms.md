# Using LLMs

## Concept

Picking the proper Large Language Model (LLM) is one of the first steps you need to consider when building any LLM application over your data.

LLMs are a core component of LlamaIndex. They can be used as standalone modules or plugged into other core LlamaIndex modules (indices, retrievers, query engines). They are always used during the response synthesis step (e.g. after retrieval). Depending on the type of index being used, LLMs may also be used during index construction, insertion, and query traversal.

LlamaIndex provides a unified interface for defining LLM modules, whether it's from OpenAI, Hugging Face, or LangChain, so that you
don't have to write the boilerplate code of defining the LLM interface yourself. This interface consists of the following (more details below):

- Support for **text completion** and **chat** endpoints (details below)
- Support for **streaming** and **non-streaming** endpoints
- Support for **synchronous** and **asynchronous** endpoints

## Usage Pattern

The following code snippet shows how you can get started using LLMs.

If you don't already have it, install your LLM:

```
pip install llama-index-llms-openai
```

Then:

```python
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

# changing the global default
Settings.llm = OpenAI()

# local usage
resp = OpenAI().complete("Paul Graham is ")
print(resp)

# per-query/chat engine
query_engine = index.as_query_engine(..., llm=llm)
chat_engine = index.as_chat_engine(..., llm=llm)
```

Find more details on [standalone usage](./llms/usage_standalone.md) or [custom usage](./llms/usage_custom.md).

## A Note on Tokenization

By default, LlamaIndex uses a global tokenizer for all token counting. This defaults to `cl100k` from tiktoken, which is the tokenizer to match the default LLM `gpt-3.5-turbo`.

If you change the LLM, you may need to update this tokenizer to ensure accurate token counts, chunking, and prompting.

The single requirement for a tokenizer is that it is a callable function, that takes a string, and returns a list.

You can set a global tokenizer like so:

```python
from llama_index.core import Settings

# tiktoken
import tiktoken

Settings.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo").encode

# huggingface
from transformers import AutoTokenizer

Settings.tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceH4/zephyr-7b-beta"
)
```

## Modules

We support integrations with OpenAI, HuggingFace, Anthropic, and more.

See the full [list of modules](./llms/modules.md).

## Further reading

- [Embeddings](./embeddings.md)
- [Prompts](./prompts/index.md)
- [Local LLMs](./llms/local.md)
- [Running Llama2 Locally](https://replicate.com/blog/run-llama-locally)
