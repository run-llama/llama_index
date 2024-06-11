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
llm = AI21(model="jamba-instruct", api_key=api_key)
```

### Chat Completions

```python
from llama_index.llms.ai21 import AI21
from llama_index.core.base.llms.types import ChatMessage

api_key = "your_api_key"
llm = AI21(model="jamba-instruct", api_key=api_key)

messages = [ChatMessage(role="user", content="What is the meaning of life?")]
response = llm.chat(messages)
print(response.message.content)
```

### Chat Streaming

```python
from llama_index.llms.ai21 import AI21
from llama_index.core.base.llms.types import ChatMessage

api_key = "your_api_key"
llm = AI21(model="jamba-instruct", api_key=api_key)

messages = [ChatMessage(role="user", content="What is the meaning of life?")]

for chunk in llm.stream_chat(messages):
    print(chunk.message.content)
```

### Text Completion

```python
from llama_index.llms.ai21 import AI21

api_key = "your_api_key"
llm = AI21(model="jamba-instruct", api_key=api_key)

response = llm.complete(prompt="What is the meaning of life?")
print(response.text)
```

### Stream Text Completion

```python
from llama_index.llms.ai21 import AI21

api_key = "your_api_key"
llm = AI21(model="jamba-instruct", api_key=api_key)

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
llm = AI21(model="jamba-instruct", api_key=api_key)
tokenizer = llm.tokenizer

tokens = tokenizer.encode("What is the meaning of life?")
print(tokens)

text = tokenizer.decode(tokens)
print(text)
```
