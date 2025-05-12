# LlamaIndex LLMs Integration: Llama API from Meta

## Installation

1. Install the required Python packages

   ```bash
   pip install llama-index-llms-meta
   pip install llama-index
   ```

2. Get API Key from [llama-api](https://llama.developer.meta.com?utm_source=partner-llamaindex&utm_medium=readme)

   ```bash
   export LLAMA_API_KEY=your_api_key
   ```

## Usage

### Basic Chat

To simulate a chat with multiple messages:

```python
from llama_index.core.llms import ChatMessage
from llama_index.llms.meta import LlamaLLM

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = LlamaLLM(
    model="Llama-3.3-8B-Instruct", api_key=os.environ["LLAMA_API_KEY"]
).chat(messages)
print(resp)
```

Example output (partial):

```
assistant: Yer lookin' fer me name, eh? Well, matey, me name be Captain Zephyr "Blackheart" McScurvy!
```

### Streaming Chat

For a streamed conversation, use `stream_chat`:

```python
from llama_index.core.llms import ChatMessage
from llama_index.llms.meta import LlamaLLM

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]

resp = LlamaLLM(
    model="Llama-3.3-8B-Instruct", api_key=os.environ["LLAMA_API_KEY"]
).stream_chat(messages)

for r in resp:
    print(r.delta, end="")
```

Example output (partial):

```
Yer lookin' fer me name, eh? Well, matey, me name be Captain Zephyr "Blackheart" McScurvy!
```
