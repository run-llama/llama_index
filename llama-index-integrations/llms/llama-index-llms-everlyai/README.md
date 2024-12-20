# LlamaIndex Llms Integration: Everlyai

## Installation

1. Install the required Python packages:

   ```bash
   %pip install llama-index-llms-everlyai
   !pip install llama-index
   ```

2. Set the EverlyAI API key as an environment variable or pass it directly to the constructor:

   ```python
   import os

   os.environ["EVERLYAI_API_KEY"] = "<your-api-key>"
   ```

   Or use it directly in your Python code:

   ```python
   llm = EverlyAI(api_key="your-api-key")
   ```

## Usage

### Basic Chat

To send a message and get a response (e.g., a joke):

```python
from llama_index.llms.everlyai import EverlyAI
from llama_index.core.llms import ChatMessage

# Initialize EverlyAI with API key
llm = EverlyAI(api_key="your-api-key")

# Create a message
message = ChatMessage(role="user", content="Tell me a joke")

# Call the chat method
resp = llm.chat([message])
print(resp)
```

Example output:

```
Why don't scientists trust atoms?
Because they make up everything!
```

### Streamed Chat

To stream a response for more dynamic conversations (e.g., storytelling):

```python
message = ChatMessage(role="user", content="Tell me a story in 250 words")
resp = llm.stream_chat([message])

for r in resp:
    print(r.delta, end="")
```

Example output (partial):

```
As the sun set over the horizon, a young girl named Lily sat on the beach, watching the waves roll in...
```

### Complete Tasks

To use the `complete` method for simpler tasks like telling a joke:

```python
resp = llm.complete("Tell me a joke")
print(resp)
```

Example output:

```
Why don't scientists trust atoms?
Because they make up everything!
```

### Streamed Completion

For generating responses like stories using `stream_complete`:

```python
resp = llm.stream_complete("Tell me a story in 250 words")

for r in resp:
    print(r.delta, end="")
```

Example output (partial):

```
As the sun set over the horizon, a young girl named Maria sat on the beach, watching the waves roll in...
```

## Notes

- Ensure the API key is set correctly before making any requests.
- The `stream_chat` and `stream_complete` methods allow for real-time response streaming, making them ideal for dynamic and lengthy outputs like stories.

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/everlyai/
