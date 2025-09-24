# Using LLMs as standalone modules

You can use our LLM modules on their own.

## Text Completion Example

```python
from llama_index.llms.openai import OpenAI

# non-streaming
completion = OpenAI().complete("Paul Graham is ")
print(completion)

# using streaming endpoint
from llama_index.llms.openai import OpenAI

llm = OpenAI()
completions = llm.stream_complete("Paul Graham is ")
for completion in completions:
    print(completion.delta, end="")
```

## Chat Example

```python
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = OpenAI().chat(messages)
print(resp)
```

Check out our [modules section](/python/framework/module_guides/models/llms/modules) for usage guides for each LLM.
