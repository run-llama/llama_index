# LlamaIndex Llms Integration: Cerebras

At Cerebras, we've developed the world's largest and fastest AI processor, the Wafer-Scale Engine-3 (WSE-3). The Cerebras CS-3 system, powered by the WSE-3, represents a new class of AI supercomputer that sets the standard for generative AI training and inference with unparalleled performance and scalability.

With Cerebras as your inference provider, you can:

- Achieve unprecedented speed for AI inference workloads
- Build commercially with high throughput
- Effortlessly scale your AI workloads with our seamless clustering technology

Our CS-3 systems can be quickly and easily clustered to create the largest AI supercomputers in the world, making it simple to place and run the largest models. Leading corporations, research institutions, and governments are already using Cerebras solutions to develop proprietary models and train popular open-source models.

Want to experience the power of Cerebras? Check out our [website](https://cerebras.net) for more resources and explore options for accessing our technology through the Cerebras Cloud or on-premise deployments!

For more information about Cerebras Cloud, visit [cloud.cerebras.ai](https://cloud.cerebras.ai/). Our API reference is available at [inference-docs.cerebras.ai](https://inference-docs.cerebras.ai/).

## Installation

using poetry:

```shell
poetry add llama-index-llms-cerebras
```

or using pip:

```shell
pip install llama-index-llms-cerebras
```

## Basic Usage

Get an API Key from [cloud.cerebras.ai](https://cloud.cerebras.ai/) and add it to your environment variables:

```
export CEREBRAS_API_KEY=<your api key>
```

Then try out one of these examples:

```python
import os

from llama_index.core.llms import ChatMessage
from llama_index.llms.cerebras import Cerebras

llm = Cerebras(model="llama-3.3-70b", api_key=os.environ["CEREBRAS_API_KEY"])

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.chat(messages)
print(resp)
```

Or alternatively with streaming:

```python
import os

from llama_index.llms.cerebras import Cerebras

llm = Cerebras(model="llama-3.3-70b", api_key=os.environ["CEREBRAS_API_KEY"])

response = llm.stream_complete("What is Generative AI?")
for r in response:
    print(r.delta, end="")
```
