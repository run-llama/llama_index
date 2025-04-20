# LlamaIndex Llms Integration: Perplexity

The Perplexity integration for LlamaIndex allows you to tap into real-time generative search powered by the Perplexity API. This integration supports synchronous and asynchronous chat completions—as well as streaming responses.

## Installation

To install the required packages, run:

```bash
%pip install llama-index-llms-perplexity
!pip install llama-index
```

## Setup

### Import Libraries and Configure API Key

Please refer to the official Perplexity [API documentation](https://docs.perplexity.ai/home) to get started. You can follow the steps outlined [here](https://docs.perplexity.ai/guides/getting-started) to generate your API key.

Import the necessary libraries and set your Perplexity API key:

```python
from llama_index.llms.perplexity import Perplexity

pplx_api_key = "your-perplexity-api-key"  # Replace with your actual API key
```

### Initialize the Perplexity LLM

Create an instance of the Perplexity LLM with your API key and desired model settings:

```python
llm = Perplexity(api_key=pplx_api_key, model="sonar-pro", temperature=0.2)
```

## Chat Example

### Sending a Chat Message

You can send a chat message using the `chat` method. Here’s how to do that:

```python
from llama_index.core.llms import ChatMessage

messages_dict = [
    {"role": "system", "content": "Be precise and concise."},
    {
        "role": "user",
        "content": "What is the weather like in San Francisco today?",
    },
]

messages = [ChatMessage(**msg) for msg in messages_dict]

# Obtain a response from the model
response = llm.chat(messages)
print(response)
```

### Async Chat

For asynchronous conversation processing, use the `achat` method to send messages and await the response:

```python
response = await llm.achat(messages)
print(response)
```

### Stream Chat

For cases where you want to receive a response token by token in real time, use the `stream_chat` method:

```python
resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")
```

### Async Stream Chat

Similarly, for asynchronous streaming, the `astream_chat` method provides a way to process response deltas asynchronously:

```python
resp = await llm.astream_chat(messages)
async for delta in resp:
    print(delta.delta, end="")
```

### Tool calling

Perplexity models can easily be wrapped into a llamaindex tool so that it can be called as part of your data processing or conversational workflows. This tool uses real-time generative search powered by Perplexity, and it’s configured with the updated default model ("sonar-pro") and the enable_search_classifier parameter enabled.

Below is an example of how to define and register the tool:

```python
from llama_index.core.tools import FunctionTool
from llama_index.llms.perplexity import Perplexity
from llama_index.core.llms import ChatMessage


def query_perplexity(query: str) -> str:
    """
    Queries the Perplexity API via the LlamaIndex integration.

    This function instantiates a Perplexity LLM with updated default settings
    (using model "sonar-pro" and enabling search classifier so that the API can
    intelligently decide if a search is needed), wraps the query into a ChatMessage,
    and returns the generated response content.
    """
    pplx_api_key = (
        "your-perplexity-api-key"  # Replace with your actual API key
    )

    llm = Perplexity(
        api_key=pplx_api_key,
        model="sonar-pro",
        temperature=0.7,
        enable_search_classifier=True,  # This will determine if the search component is necessary in this particular context
    )

    messages = [ChatMessage(role="user", content=query)]
    response = llm.chat(messages)
    return response.message.content


# Create the tool from the query_perplexity function
query_perplexity_tool = FunctionTool.from_defaults(fn=query_perplexity)
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/perplexity/
