# NVIDIA's LLM connector

Install the connector,

```shell
pip install llama-index-llms-nvidia
```

With this connector, you'll be able to connect to and generate from compatible models available as hosted [NVIDIA NIMs](https://ai.nvidia.com), such as:

- Google's [gemma-7b](https://build.nvidia.com/google/gemma-7b)
- Mistal AI's [mistral-7b-instruct-v0.2](https://build.nvidia.com/mistralai/mistral-7b-instruct-v2)
- And more!

_First_, get a free API key. Go to https://build.nvidia.com, select a model, click "Get API Key".
Store this key in your environment as `NVIDIA_API_KEY`.

_Then_, try it out.

```python
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.llms import ChatMessage, MessageRole

llm = NVIDIA()

messages = [
    ChatMessage(
        role=MessageRole.SYSTEM, content=("You are a helpful assistant.")
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=("What are the most popular house pets in North America?"),
    ),
]

llm.chat(messages)
```
