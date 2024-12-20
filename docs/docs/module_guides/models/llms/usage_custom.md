# Customizing LLMs within LlamaIndex Abstractions

You can plugin these LLM abstractions within our other modules in LlamaIndex (indexes, retrievers, query engines, agents) which allow you to build advanced workflows over your data.

By default, we use OpenAI's `gpt-3.5-turbo` model. But you may choose to customize
the underlying LLM being used.

Below we show a few examples of LLM customization. This includes

- changing the underlying LLM
- changing the number of output tokens (for OpenAI, Cohere, or AI21)
- having more fine-grained control over all parameters for any LLM, from context window to chunk overlap

## Example: Changing the underlying LLM

An example snippet of customizing the LLM being used is shown below.
In this example, we use `gpt-4` instead of `gpt-3.5-turbo`. Available models include `gpt-3.5-turbo`, `gpt-3.5-turbo-instruct`, `gpt-3.5-turbo-16k`, `gpt-4`, `gpt-4-32k`, `text-davinci-003`, and `text-davinci-002`.

Note that
you may also plug in any LLM shown on Langchain's
[LLM](https://python.langchain.com/docs/integrations/llms/) page.

```python
from llama_index.core import KeywordTableIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# alternatively
# from langchain.llms import ...

documents = SimpleDirectoryReader("data").load_data()

# define LLM
llm = OpenAI(temperature=0.1, model="gpt-4")

# build index
index = KeywordTableIndex.from_documents(documents, llm=llm)

# get response from query
query_engine = index.as_query_engine()
response = query_engine.query(
    "What did the author do after his time at Y Combinator?"
)
```

## Example: Changing the number of output tokens (for OpenAI, Cohere, AI21)

The number of output tokens is usually set to some low number by default (for instance,
with OpenAI the default is 256).

For OpenAI, Cohere, AI21, you just need to set the `max_tokens` parameter
(or maxTokens for AI21). We will handle text chunking/calculations under the hood.

```python
from llama_index.core import KeywordTableIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

documents = SimpleDirectoryReader("data").load_data()

# define global LLM
Settings.llm = OpenAI(temperature=0, model="gpt-3.5-turbo", max_tokens=512)
```

## Example: Explicitly configure `context_window` and `num_output`

If you are using other LLM classes from langchain, you may need to explicitly configure the `context_window` and `num_output` via the `Settings` since the information is not available by default.

```python
from llama_index.core import KeywordTableIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

documents = SimpleDirectoryReader("data").load_data()


# set context window
Settings.context_window = 4096
# set number of output tokens
Settings.num_output = 256

# define LLM
Settings.llm = OpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
    max_tokens=num_output,
)
```

## Example: Using a HuggingFace LLM

LlamaIndex supports using LLMs from HuggingFace directly. Note that for a completely private experience, also setup a [local embeddings model](../embeddings.md).

Many open-source models from HuggingFace require either some preamble before each prompt, which is a `system_prompt`. Additionally, queries themselves may need an additional wrapper around the `query_str` itself. All this information is usually available from the HuggingFace model card for the model you are using.

Below, this example uses both the `system_prompt` and `query_wrapper_prompt`, using specific prompts from the model card found [here](https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b).

```python
from llama_index.core import PromptTemplate


# Transform a string into input zephyr-specific input
def completion_to_prompt(completion):
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"


# Transform a list of chat messages into zephyr-specific input
def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == "user":
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == "assistant":
            prompt += f"<|assistant|>\n{message.content}</s>\n"

    # ensure we start with a system prompt, insert blank if needed
    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt

    # add final assistant prompt
    prompt = prompt + "<|assistant|>\n"

    return prompt


import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings

Settings.llm = HuggingFaceLLM(
    model_name="HuggingFaceH4/zephyr-7b-beta",
    tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
    context_window=3900,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    device_map="auto",
)
```

Some models will raise errors if all the keys from the tokenizer are passed to the model. A common tokenizer output that causes issues is `token_type_ids`. Below is an example of configuring the predictor to remove this before passing the inputs to the model:

```python
HuggingFaceLLM(
    # ...
    tokenizer_outputs_to_remove=["token_type_ids"]
)
```

A full API reference can be found [here](../../../api_reference/llms/huggingface.md).

Several example notebooks are also listed below:

- [StableLM](../../../examples/customization/llms/SimpleIndexDemo-Huggingface_stablelm.ipynb)
- [Camel](../../../examples/customization/llms/SimpleIndexDemo-Huggingface_camel.ipynb)

## Example: Using a Custom LLM Model - Advanced

To use a custom LLM model, you only need to implement the `LLM` class (or `CustomLLM` for a simpler interface)
You will be responsible for passing the text to the model and returning the newly generated tokens.

This implementation could be some local model, or even a wrapper around your own API.

Note that for a completely private experience, also setup a [local embeddings model](../embeddings.md).

Here is a small boilerplate example:

```python
from typing import Optional, List, Mapping, Any

from llama_index.core import SimpleDirectoryReader, SummaryIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import Settings


class OurLLM(CustomLLM):
    context_window: int = 3900
    num_output: int = 256
    model_name: str = "custom"
    dummy_response: str = "My response"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=self.dummy_response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)


# define our LLM
Settings.llm = OurLLM()

# define embed model
Settings.embed_model = "local:BAAI/bge-base-en-v1.5"


# Load the your data
documents = SimpleDirectoryReader("./data").load_data()
index = SummaryIndex.from_documents(documents)

# Query and print response
query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")
print(response)
```

Using this method, you can use any LLM. Maybe you have one running locally, or running on your own server. As long as the class is implemented and the generated tokens are returned, it should work out. Note that we need to use the prompt helper to customize the prompt sizes, since every model has a slightly different context length.

The decorator is optional, but provides observability via callbacks on the LLM calls.

Note that you may have to adjust the internal prompts to get good performance. Even then, you should be using a sufficiently large LLM to ensure it's capable of handling the complex queries that LlamaIndex uses internally, so your mileage may vary.

A list of all default internal prompts is available [here](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/prompts/default_prompts.py), and chat-specific prompts are listed [here](https://github.com/run-llama/llama_index/blob/main/llama_index/prompts/chat_prompts.py). You can also implement [your own custom prompts](../prompts/index.md).
