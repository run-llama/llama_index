# LlamaIndex Llms Integration: Mistral

## Installation

Install the required packages using the following commands:

```bash
%pip install llama-index-llms-mistralai
!pip install llama-index
```

## Basic Usage

### Initialize the MistralAI Model

To use the MistralAI model, create an instance and provide your API key:

```python
from llama_index.llms.mistralai import MistralAI

llm = MistralAI(api_key="<replace-with-your-key>")
```

### Generate Completions

To generate a text completion for a prompt, use the `complete` method:

```python
resp = llm.complete("Paul Graham is ")
print(resp)
```

### Chat with the Model

You can also chat with the model using a list of messages. Here’s an example:

```python
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(role="system", content="You are CEO of MistralAI."),
    ChatMessage(role="user", content="Tell me the story about La plateforme"),
]
resp = MistralAI().chat(messages)
print(resp)
```

### Using Random Seed

To set a random seed for reproducibility, initialize the model with the `random_seed` parameter:

```python
resp = MistralAI(random_seed=42).chat(messages)
print(resp)
```

## Streaming Responses

### Stream Completions

You can stream responses using the `stream_complete` method:

```python
resp = llm.stream_complete("Paul Graham is ")
for r in resp:
    print(r.delta, end="")
```

### Stream Chat Responses

To stream chat messages, use the following code:

```python
messages = [
    ChatMessage(role="system", content="You are CEO of MistralAI."),
    ChatMessage(role="user", content="Tell me the story about La plateforme"),
]
resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")
```

## Configure Model

To use a specific model configuration, initialize the model with the desired model name:

```python
llm = MistralAI(model="mistral-medium")
resp = llm.stream_complete("Paul Graham is ")
for r in resp:
    print(r.delta, end="")
```

## Function Calling

You can call functions from the model by defining tools. Here’s an example:

```python
from llama_index.llms.mistralai import MistralAI
from llama_index.core.tools import FunctionTool


def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


def mystery(a: int, b: int) -> int:
    """Mystery function on two integers."""
    return a * b + a + b


mystery_tool = FunctionTool.from_defaults(fn=mystery)
multiply_tool = FunctionTool.from_defaults(fn=multiply)

llm = MistralAI(model="mistral-large-latest")
response = llm.predict_and_call(
    [mystery_tool, multiply_tool],
    user_msg="What happens if I run the mystery function on 5 and 7",
)
print(str(response))
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/mistralai/
