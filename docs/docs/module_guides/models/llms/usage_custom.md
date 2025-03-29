# Customizing LLMs within LlamaIndex Abstractions

You can plugin these LLM abstractions within our other modules in LlamaIndex (indexes, retrievers, query engines, agents) which allow you to build advanced workflows over your data.

By default, we use OpenAI's `gpt-3.5-turbo` model. But you may choose to customize
the underlying LLM being used.

## Example: Changing the underlying LLM

An example snippet of customizing the LLM being used is shown below.

In this example, we use `gpt-4o-mini` instead of `gpt-3.5-turbo`. Available models include `gpt-4o-mini`, `gpt-4o`, `o3-mini`, and more.

```python
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# define LLM
llm = OpenAI(temperature=0.1, model="gpt-4o-mini")

# change the global default LLM
Settings.llm = llm

documents = SimpleDirectoryReader("data").load_data()

# build index
index = VectorStoreIndex.from_documents(documents)

# locally override the LLM
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query(
    "What did the author do after his time at Y Combinator?"
)
```

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
