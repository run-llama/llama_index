# LlamaIndex Llms Integration: OctoAI

Using the [OctoAI](https://octo.ai) LLMs Integration is a simple as:

```python
from llama_index.llms.octoai import OctoAI

octoai = OctoAI(token=OCTOAI_API_KEY)
response = octoai.complete("Paul Graham is ")
print(response)
```

## Calling `chat` with a list of messages

```python
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(
        role="system",
        content="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
    ),
    ChatMessage(role="user", content="Write a blog about Seattle"),
]
response = octoai.chat(messages)
print(response)
```
