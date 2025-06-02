# LlamaIndex Llms Integration: Friendli

## Installation

1. Install the required Python packages:

   ```bash
   %pip install llama-index-llms-friendli
   !pip install llama-index
   ```

2. Set the Friendli token as an environment variable:

   ```bash
   %env FRIENDLI_TOKEN=your_token_here
   ```

## Usage

### Basic Chat

To generate a chat response, use the following code:

```python
from llama_index.llms.friendli import Friendli
from llama_index.core.llms import ChatMessage, MessageRole

llm = Friendli()

message = ChatMessage(role=MessageRole.USER, content="Tell me a joke.")
resp = llm.chat([message])
print(resp)
```

### Streaming Responses

To stream chat responses in real-time:

```python
resp = llm.stream_chat([message])
for r in resp:
    print(r.delta, end="")
```

### Asynchronous Chat

For asynchronous chat interactions, use the following:

```python
resp = await llm.achat([message])
print(resp)
```

### Async Streaming

To handle async streaming of chat responses:

```python
resp = await llm.astream_chat([message])
async for r in resp:
    print(r.delta, end="")
```

### Complete with a Prompt

To generate a completion based on a prompt:

```python
prompt = "Draft a cover letter for a role in software engineering."
resp = llm.complete(prompt)
print(resp)
```

### Streaming Completion

To stream completions in real-time:

```python
resp = llm.stream_complete(prompt)
for r in resp:
    print(r.delta, end="")
```

### Async Completion

To handle async completions:

```python
resp = await llm.acomplete(prompt)
print(resp)
```

### Async Streaming Completion

For async streaming of completions:

```python
resp = await llm.astream_complete(prompt)
async for r in resp:
    print(r.delta, end="")
```

### Model Configuration

To configure a specific model:

```python
llm = Friendli(model="llama-2-70b-chat")
resp = llm.chat([message])
print(resp)
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/friendli/
