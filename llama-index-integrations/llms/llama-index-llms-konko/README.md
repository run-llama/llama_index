# LlamaIndex Llms Integration: Konko

## Installation

1. Install the required Python packages:

   ```bash
   %pip install llama-index-llms-konko
   !pip install llama-index
   ```

2. Set the API keys as environment variables:

   ```bash
   export KONKO_API_KEY=<your-api-key>
   export OPENAI_API_KEY=<your-api-key>
   ```

## Usage

### Import Required Libraries

```python
import os
from llama_index.llms.konko import Konko
from llama_index.core.llms import ChatMessage
```

### Chat with Konko Model

To chat with a Konko model:

```python
os.environ["KONKO_API_KEY"] = "<your-api-key>"
llm = Konko(model="meta-llama/llama-2-13b-chat")
messages = ChatMessage(role="user", content="Explain Big Bang Theory briefly")

resp = llm.chat([messages])
print(resp)
```

### Chat with OpenAI Model

To chat with an OpenAI model:

```python
os.environ["OPENAI_API_KEY"] = "<your-api-key>"
llm = Konko(model="gpt-3.5-turbo")
message = ChatMessage(role="user", content="Explain Big Bang Theory briefly")

resp = llm.chat([message])
print(resp)
```

### Streaming Responses

To stream a response for longer messages:

```python
message = ChatMessage(role="user", content="Tell me a story in 250 words")
resp = llm.stream_chat([message], max_tokens=1000)

for r in resp:
    print(r.delta, end="")
```

### Complete with Prompt

To generate a completion based on a system prompt:

```python
llm = Konko(model="phind/phind-codellama-34b-v2", max_tokens=100)
text = """### System Prompt
You are an intelligent programming assistant.

### User Message
Implement a linked list in C++

### Assistant
..."""

resp = llm.stream_complete(text, max_tokens=1000)
for r in resp:
    print(r.delta, end="")
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/konko/
