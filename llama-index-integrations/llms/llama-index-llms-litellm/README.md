# LlamaIndex Llms Integration: Litellm

## Installation

1. Install the required Python packages:

   ```bash
   %pip install llama-index-llms-litellm
   !pip install llama-index
   ```

## Usage

### Import Required Libraries

```python
import os
from llama_index.llms.litellm import LiteLLM
from llama_index.core.llms import ChatMessage
```

### Set Up Environment Variables

Set your API keys as environment variables:

```python
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["COHERE_API_KEY"] = "your-api-key"
```

### Example: OpenAI Call

To interact with the OpenAI model:

```python
message = ChatMessage(role="user", content="Hey! how's it going?")
llm = LiteLLM("gpt-3.5-turbo")
chat_response = llm.chat([message])
print(chat_response)
```

### Example: Cohere Call

To interact with the Cohere model:

```python
llm = LiteLLM("command-nightly")
chat_response = llm.chat([message])
print(chat_response)
```

### Example: Chat with System Message

To have a chat with a system role:

```python
messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]
resp = LiteLLM("gpt-3.5-turbo").chat(messages)
print(resp)
```

### Streaming Responses

To use the streaming feature with `stream_complete`:

```python
llm = LiteLLM("gpt-3.5-turbo")
resp = llm.stream_complete("Paul Graham is ")
for r in resp:
    print(r.delta, end="")
```

### Streaming Chat Example

To stream chat messages:

```python
llm = LiteLLM("gpt-3.5-turbo")
resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")
```

### Asynchronous Example

For asynchronous calls, use:

```python
llm = LiteLLM("gpt-3.5-turbo")
resp = await llm.acomplete("Paul Graham is ")
print(resp)
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/litellm/
