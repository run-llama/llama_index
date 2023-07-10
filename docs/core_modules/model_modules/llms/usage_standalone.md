# Using LLMs as standalone modules

You can use our LLM modules on their own.

## Text Completion Example

```python
from llama_index.llms import OpenAI

# non-streaming
resp = OpenAI().complete('Paul Graham is ')
print(resp)

# using streaming endpoint
from llama_index.llms import OpenAI
llm = OpenAI()
resp = llm.stream_complete('Paul Graham is ')
for delta in resp:
    print(delta, end='')
```

## Chat Example

```python
from llama_index.llms import ChatMessage, OpenAI

messages = [
    ChatMessage(role="system", content="You are a pirate with a colorful personality"),
    ChatMessage(role="user", content="What is your name"),
]
resp = OpenAI().chat(messages)
print(resp)
```

Check out our [modules section](modules.md) for usage guides for each LLM.
