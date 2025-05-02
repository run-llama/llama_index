# LlamaIndex LLMs Integration: AI21 Labs

## Installation

First, you need to install the package. You can do this using pip:

```bash
pip install llama-index-llms-ai21
```

## Usage

Here's a basic example of how to use the AI21 class to generate text completions and handle chat interactions.

## Initializing the AI21 Client

You need to initialize the AI21 client with the appropriate model and API key.

```python
from llama_index.llms.ai21 import AI21

api_key = "your_api_key"
llm = AI21(model="jamba-1.5-mini", api_key=api_key)
```

### Chat Completions

```python
from llama_index.llms.ai21 import AI21
from llama_index.core.base.llms.types import ChatMessage

api_key = "your_api_key"
llm = AI21(model="jamba-1.5-mini", api_key=api_key)

messages = [ChatMessage(role="user", content="What is the meaning of life?")]
response = llm.chat(messages)
print(response.message.content)
```

### Chat Streaming

```python
from llama_index.llms.ai21 import AI21
from llama_index.core.base.llms.types import ChatMessage

api_key = "your_api_key"
llm = AI21(model="jamba-1.5-mini", api_key=api_key)

messages = [ChatMessage(role="user", content="What is the meaning of life?")]

for chunk in llm.stream_chat(messages):
    print(chunk.message.content)
```

### Text Completion

```python
from llama_index.llms.ai21 import AI21

api_key = "your_api_key"
llm = AI21(model="jamba-1.5-mini", api_key=api_key)

response = llm.complete(prompt="What is the meaning of life?")
print(response.text)
```

### Stream Text Completion

```python
from llama_index.llms.ai21 import AI21

api_key = "your_api_key"
llm = AI21(model="jamba-1.5-mini", api_key=api_key)

response = llm.stream_complete(prompt="What is the meaning of life?")

for chunk in response:
    print(response.text)
```

## Other Models Support

You could also use more model types. For example the `j2-ultra` and `j2-mid`

These models support `chat` and `complete` methods only.

### Chat

```python
from llama_index.llms.ai21 import AI21
from llama_index.core.base.llms.types import ChatMessage

api_key = "your_api_key"
llm = AI21(model="j2-chat", api_key=api_key)

messages = [ChatMessage(role="user", content="What is the meaning of life?")]
response = llm.chat(messages)
print(response.message.content)
```

### Complete

```python
from llama_index.llms.ai21 import AI21

api_key = "your_api_key"
llm = AI21(model="j2-ultra", api_key=api_key)

response = llm.complete(prompt="What is the meaning of life?")
print(response.text)
```

## Tokenizer

The type of the tokenizer is determined by the name of the model

```python
from llama_index.llms.ai21 import AI21

api_key = "your_api_key"
llm = AI21(model="jamba-1.5-mini", api_key=api_key)
tokenizer = llm.tokenizer

tokens = tokenizer.encode("What is the meaning of life?")
print(tokens)

text = tokenizer.decode(tokens)
print(text)
```

## Async Support

You can also use the async functionalities

### async chat

```python
from llama_index.llms.ai21 import AI21
from llama_index.core.base.llms.types import ChatMessage


async def main():
    api_key = "your_api_key"
    llm = AI21(model="jamba-1.5-mini", api_key=api_key)

    messages = [
        ChatMessage(role="user", content="What is the meaning of life?")
    ]
    response = await llm.achat(messages)
    print(response.message.content)
```

### async stream_chat

```python
from llama_index.llms.ai21 import AI21
from llama_index.core.base.llms.types import ChatMessage


async def main():
    api_key = "your_api_key"
    llm = AI21(model="jamba-1.5-mini", api_key=api_key)

    messages = [
        ChatMessage(role="user", content="What is the meaning of life?")
    ]
    response = await llm.astream_chat(messages)

    async for chunk in response:
        print(chunk.message.content)
```

## Tool Calling

```python
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.llms.ai21 import AI21
from llama_index.core.tools import FunctionTool


def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


def subtract(a: int, b: int) -> int:
    """Subtract two integers and returns the result integer"""
    return a - b


def divide(a: int, b: int) -> float:
    """Divide two integers and returns the result float"""
    return a - b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
divide_tool = FunctionTool.from_defaults(fn=divide)

api_key = "your_api_key"

llm = AI21(model="jamba-1.5-mini", api_key=api_key)

agent_worker = FunctionCallingAgentWorker.from_tools(
    [multiply_tool, add_tool, subtract_tool, divide_tool],
    llm=llm,
    verbose=True,
    allow_parallel_tool_calls=True,
)
agent = agent_worker.as_agent()

response = agent.chat(
    "My friend Moses had 10 apples. He ate 5 apples in the morning. Then he found a box with 25 apples."
    "He divided all his apples between his 5 friends. How many apples did each friend get?"
)
```
