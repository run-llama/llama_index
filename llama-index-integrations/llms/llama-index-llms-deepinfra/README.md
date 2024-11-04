# LlamaIndex Llms Integration: DeepInfra

## Installation

First, install the necessary package:

```bash
pip install llama-index-llms-deepinfra
```

## Initialization

Set up the `DeepInfraLLM` class with your API key and desired parameters:

```python
from llama_index.llms.deepinfra import DeepInfraLLM
import asyncio

llm = DeepInfraLLM(
    model="mistralai/Mixtral-8x22B-Instruct-v0.1",  # Default model name
    api_key="your-deepinfra-api-key",  # Replace with your DeepInfra API key
    temperature=0.5,
    max_tokens=50,
    additional_kwargs={"top_p": 0.9},
)
```

## Synchronous Complete

Generate a text completion synchronously using the `complete` method:

```python
response = llm.complete("Hello World!")
print(response.text)
```

## Synchronous Stream Complete

Generate a streaming text completion synchronously using the `stream_complete` method:

```python
content = ""
for completion in llm.stream_complete("Once upon a time"):
    content += completion.delta
    print(completion.delta, end="")
```

## Synchronous Chat

Generate a chat response synchronously using the `chat` method:

```python
from llama_index.core.base.llms.types import ChatMessage

messages = [
    ChatMessage(role="user", content="Tell me a joke."),
]
chat_response = llm.chat(messages)
print(chat_response.message.content)
```

## Synchronous Stream Chat

Generate a streaming chat response synchronously using the `stream_chat` method:

```python
messages = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="Tell me a story."),
]
content = ""
for chat_response in llm.stream_chat(messages):
    content += chat_response.message.delta
    print(chat_response.message.delta, end="")
```

## Asynchronous Complete

Generate a text completion asynchronously using the `acomplete` method:

```python
async def async_complete():
    response = await llm.acomplete("Hello Async World!")
    print(response.text)


asyncio.run(async_complete())
```

## Asynchronous Stream Complete

Generate a streaming text completion asynchronously using the `astream_complete` method:

```python
async def async_stream_complete():
    content = ""
    response = await llm.astream_complete("Once upon an async time")
    async for completion in response:
        content += completion.delta
        print(completion.delta, end="")


asyncio.run(async_stream_complete())
```

## Asynchronous Chat

Generate a chat response asynchronously using the `achat` method:

```python
async def async_chat():
    messages = [
        ChatMessage(role="user", content="Tell me an async joke."),
    ]
    chat_response = await llm.achat(messages)
    print(chat_response.message.content)


asyncio.run(async_chat())
```

## Asynchronous Stream Chat

Generate a streaming chat response asynchronously using the `astream_chat` method:

```python
async def async_stream_chat():
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Tell me an async story."),
    ]
    content = ""
    response = await llm.astream_chat(messages)
    async for chat_response in response:
        content += chat_response.message.delta
        print(chat_response.message.delta, end="")


asyncio.run(async_stream_chat())
```

---

For any questions or feedback, please contact us at [feedback@deepinfra.com](mailto:feedback@deepinfra.com).
