# LlamaIndex Llms Integration: Fireworks

## Installation

1. Install the required Python packages:

   ```bash
   %pip install llama-index-llms-fireworks
   %pip install llama-index
   ```

2. Set the Fireworks API key as an environment variable or pass it directly to the class constructor.

## Usage

### Basic Completion

To generate a simple completion, use the `complete` method:

```python
from llama_index.llms.fireworks import Fireworks

resp = Fireworks().complete("Paul Graham is ")
print(resp)
```

Example output:

```
Paul Graham is a well-known essayist, programmer, and startup entrepreneur. He co-founded Y Combinator, which supported startups like Dropbox, Airbnb, and Reddit.
```

### Basic Chat

To simulate a chat with multiple messages:

```python
from llama_index.core.llms import ChatMessage
from llama_index.llms.fireworks import Fireworks

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = Fireworks().chat(messages)
print(resp)
```

Example output:

```
Arr matey, ye be askin' for me name? Well, I be known as Captain Redbeard the Terrible!
```

### Streaming Completion

To stream a response in real-time using `stream_complete`:

```python
from llama_index.llms.fireworks import Fireworks

llm = Fireworks()
resp = llm.stream_complete("Paul Graham is ")

for r in resp:
    print(r.delta, end="")
```

Example output (partial):

```
Paul Graham is a well-known essayist, programmer, and venture capitalist...
```

### Streaming Chat

For a streamed conversation, use `stream_chat`:

```python
from llama_index.llms.fireworks import Fireworks
from llama_index.core.llms import ChatMessage

llm = Fireworks()
messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.stream_chat(messages)

for r in resp:
    print(r.delta, end="")
```

Example output (partial):

```
Arr matey, ye be askin' for me name? Well, I be known as Captain Redbeard the Terrible...
```

### Model Configuration

To configure the model for more specific behavior:

```python
from llama_index.llms.fireworks import Fireworks

llm = Fireworks(model="accounts/fireworks/models/firefunction-v1")
resp = llm.complete("Paul Graham is ")
print(resp)
```

Example output:

```
Paul Graham is an English-American computer scientist, entrepreneur, venture capitalist, and blogger.
```

### API Key Configuration

To use separate API keys for different instances:

```python
from llama_index.llms.fireworks import Fireworks

llm = Fireworks(
    model="accounts/fireworks/models/firefunction-v1", api_key="YOUR_API_KEY"
)
resp = llm.complete("Paul Graham is ")
print(resp)
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/fireworks/
