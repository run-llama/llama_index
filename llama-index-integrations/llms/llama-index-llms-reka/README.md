# LlamaIndex Llms Integration: Reka

This package provides integration between the Reka language model and LlamaIndex, allowing you to use Reka's powerful language models in your LlamaIndex applications.
Installation
To use this integration, you need to install the llama-index-llms-reka package:

```bash
pip install llama-index-llms-reka
```

To obtain API key, please visit [https://platform.reka.ai/](https://platform.reka.ai/)
Our baseline models always available for public access are:

- `reka-edge`
- `reka-flash`
- `reka-core`

Other models may be available. The Get Models API allows you to list what models you have available to you. Using the Python SDK, it can be accessed as follows:

```python
from reka.client import Reka

client = Reka()
print(client.models.get())
```

Here are some examples of how to use the Reka LLM integration with LlamaIndex:

```python
import os
from llama_index.llms.reka import RekaLLM

api_key = os.getenv("REKA_API_KEY")
reka_llm = RekaLLM(model="reka-flash", api_key=api_key)
```

# Initialize the Reka LLM client

```python
api_key = os.getenv("REKA_API_KEY")
reka_llm = RekaLLM(model="reka-flash", api_key=api_key)
```

# Chat completion

```python
from llama_index.core.base.llms.types import ChatMessage, MessageRole

messages = [
    ChatMessage(
        role=MessageRole.SYSTEM, content="You are a helpful assistant."
    ),
    ChatMessage(
        role=MessageRole.USER, content="What is the capital of France?"
    ),
]
response = reka_llm.chat(messages)
print(response.message.content)
```

# Text completion

```python
prompt = "The capital of France is"
response = reka_llm.complete(prompt)
print(response.text)
```

Streaming Responses
python

# Streaming chat completion

```python
messages = [
    ChatMessage(
        role=MessageRole.SYSTEM, content="You are a helpful assistant."
    ),
    ChatMessage(
        role=MessageRole.USER,
        content="List the first 5 planets in the solar system.",
    ),
]
for chunk in reka_llm.stream_chat(messages):
    print(chunk.delta, end="", flush=True)
```

# Streaming text completion

```python
prompt = "List the first 5 planets in the solar system:"
for chunk in reka_llm.stream_complete(prompt):
    print(chunk.delta, end="", flush=True)
```

Asynchronous Usage

```
import asyncio

async def main():
    # Async chat completion
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="What is the largest planet in our solar system?"),
    ]
    response = await reka_llm.achat(messages)
    print(response.message.content)

    # Async text completion
    prompt = "The largest planet in our solar system is"
    response = await reka_llm.acomplete(prompt)
    print(response.text)

    # Async streaming chat completion
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="Name the first 5 elements in the periodic table."),
    ]
    async for chunk in await reka_llm.astream_chat(messages):
        print(chunk.delta, end="", flush=True)

    # Async streaming text completion
    prompt = "List the first 5 elements in the periodic table:"
    async for chunk in await reka_llm.astream_complete(prompt):
        print(chunk.delta, end="", flush=True)

asyncio.run(main())
```

# Running Tests

To run the tests for this integration, you'll need to have pytest and pytest-asyncio installed. You can install them using pip:

```bash
pip install pytest pytest-asyncio
```

Then, set your Reka API key as an environment variable:

```bash
export REKA_API_KEY=your_api_key_here
```

Now you can run the tests using pytest:

```bash
pytest tests/test_reka_llm.py -v
```

To run only mock integration test without remote connections
pytest tests/test_reka_llm.py -v -k "mock"
Note: The test file should be named test_reka_llm.py and placed in the appropriate directory.

# Contributing

Contributions to improve this integration are welcome. Please ensure that you add or update tests as necessary when making changes.
When adding new features or modifying existing ones, please update this README to reflect those changes.
