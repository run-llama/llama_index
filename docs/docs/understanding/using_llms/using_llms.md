# Using LLMs

!!! tip
    For a list of our supported LLMs and a comparison of their functionality, check out our [LLM module guide](../../module_guides/models/llms.md).

One of the first steps when building an LLM-based application is which LLM to use; they have different strengths and price points and you may wish to use more than one.

LlamaIndex provides a single interface to a large number of different LLMs. Using an LLM can be as simple as installing the appropriate integration:

```bash
pip install llama-index-llms-openai
```

And then calling it in a one-liner:

```python
from llama_index.llms.openai import OpenAI

response = OpenAI().complete("William Shakespeare is ")
print(response)
```

Note that this requires an API key called `OPENAI_API_KEY` in your environment; see the [starter tutorial](../../getting_started/starter_example.md) for more details.

`complete` is also available as an async method, `acomplete`.

You can also get a streaming response by calling `stream_complete`, which returns a generator that yields tokens as they are produced:

```
handle = OpenAI().stream_complete("William Shakespeare is ")

for token in handle:
    print(token.delta, end="", flush=True)
```

`stream_complete` is also available as an async method, `astream_complete`.

## Chat interface

The LLM class also implements a `chat` method, which allows you to have more sophisticated interactions:

```python
messages = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="Tell me a joke."),
]
chat_response = llm.chat(messages)
```

## Specifying models

Many LLM integrations provide more than one model. You can specify a model by passing the `model` parameter to the LLM constructor:

```python
llm = OpenAI(model="gpt-4o-mini")
response = llm.complete("Who is Laurie Voss?")
print(response)
```

## Available LLMs

We support integrations with OpenAI, Anthropic, Mistral, DeepSeek, Hugging Face, and dozens more. Check out our [module guide to LLMs](../../module_guides/models/llms.md) for a full list, including how to run a local model.

!!! tip
    A general note on privacy and LLM usage can be found on the [privacy page](./privacy.md).

### Using a local LLM

LlamaIndex doesn't just support hosted LLM APIs; you can also run a local model such as Meta's Llama 3 locally. For example, if you have [Ollama](https://github.com/ollama/ollama) installed and running:

```python
from llama_index.llms.ollama import Ollama

llm = Ollama(model="llama3.3", request_timeout=60.0)
```

See the [custom LLM's How-To](../../module_guides/models/llms/usage_custom.md) for more details on using and configuring LLM models.
