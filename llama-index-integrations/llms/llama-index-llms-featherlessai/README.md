# LlamaIndex Llms Integration: Featherless AI

[Featherless](https://featherless.ai/) is a serverless AI inference platform that makes tens of thousands of open-source AI models instantly available through a single API. Browse the catalog at [featherless.ai/models](https://featherless.ai/models).

## Installation

```shell
pip install llama-index-llms-featherlessai
```

## Setup

Get an API key from [featherless.ai/account/api-keys](https://featherless.ai/account/api-keys) and set it as an environment variable:

```shell
export FEATHERLESS_API_KEY=<your api key>
```

## Basic Usage

### Generate Completions

```python
from llama_index.llms.featherlessai import FeatherlessLLM

llm = FeatherlessLLM(model="Qwen/Qwen3.8-27B", context_window=32768)

resp = llm.complete(
    "Explain in two sentences what an inference provider does."
)
print(resp)
```

Any model id from [featherless.ai/models](https://featherless.ai/models) works as `model`. Set `context_window` to what your model and plan allow; LlamaIndex assumes 3,900 tokens otherwise.

### Chat Responses

```python
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(role="system", content="Answer in one or two sentences."),
    ChatMessage(role="user", content="What is a context window?"),
]
resp = llm.chat(messages)
print(resp)

messages.append(resp.message)
messages.append(
    ChatMessage(role="user", content="Why does that matter for agents?")
)
resp = llm.chat(messages)
print(resp)
```

### Agents and Tool Calling

```python
from datetime import datetime
from zoneinfo import ZoneInfo

from llama_index.core.agent.workflow import FunctionAgent


def get_current_time(timezone: str) -> str:
    """Get the current date and time in an IANA timezone, e.g. Asia/Tokyo."""
    return datetime.now(ZoneInfo(timezone)).strftime("%A %Y-%m-%d %H:%M")


agent = FunctionAgent(tools=[get_current_time], llm=llm)
response = await agent.run("What time is it in Tokyo right now?")
print(response)
```

See the [example notebook](https://developers.llamaindex.ai/python/framework/integrations/llm/featherlessai/) for streaming and more.
