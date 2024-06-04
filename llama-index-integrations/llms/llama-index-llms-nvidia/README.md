# NVIDIA NIMs

The `llama-index-llms-nvidia` package contains LlamaIndex integrations building applications with models on
NVIDIA NIM inference microservice. NIM supports models across domains like chat, embedding, and re-ranking models
from the community as well as NVIDIA. These models are optimized by NVIDIA to deliver the best performance on NVIDIA
accelerated infrastructure and deployed as a NIM, an easy-to-use, prebuilt containers that deploy anywhere using a single
command on NVIDIA accelerated infrastructure.

NVIDIA hosted deployments of NIMs are available to test on the [NVIDIA API catalog](https://build.nvidia.com/). After testing,
NIMs can be exported from NVIDIA’s API catalog using the NVIDIA AI Enterprise license and run on-premises or in the cloud,
giving enterprises ownership and full control of their IP and AI application.

NIMs are packaged as container images on a per model basis and are distributed as NGC container images through the NVIDIA NGC Catalog.
At their core, NIMs provide easy, consistent, and familiar APIs for running inference on an AI model.

# NVIDIA's LLM connector

This example goes over how to use LlamaIndex to interact with and develop LLM-powered systems using the publicly-accessible AI Foundation endpoints.

With this connector, you'll be able to connect to and generate from compatible models available as hosted [NVIDIA NIMs](https://ai.nvidia.com), such as:

- Google's [gemma-7b](https://build.nvidia.com/google/gemma-7b)
- Mistal AI's [mistral-7b-instruct-v0.2](https://build.nvidia.com/mistralai/mistral-7b-instruct-v2)
- And more!

## Installation

```shell
pip install llama-index-llms-nvidia
```

## Setup

**To get started:**

1. Create a free account with [NVIDIA](https://build.nvidia.com/), which hosts NVIDIA AI Foundation models.

2. Click on your model of choice.

3. Under Input select the Python tab, and click `Get API Key`. Then click `Generate Key`.

4. Copy and save the generated key as NVIDIA_API_KEY. From there, you should have access to the endpoints.

```python
import getpass
import os

if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    print("Valid NVIDIA_API_KEY already in environment. Delete to reset")
else:
    nvapi_key = getpass.getpass("NVAPI Key (starts with nvapi-): ")
    assert nvapi_key.startswith(
        "nvapi-"
    ), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvapi_key
```

## Working with API Catalog

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

## Working with NVIDIA NIMs

When ready to deploy, you can self-host models with NVIDIA NIM—which is included with the NVIDIA AI Enterprise software license—and run them anywhere, giving you ownership of your customizations and full control of your intellectual property (IP) and AI applications.

[Learn more about NIMs](https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/)

```python
from llama_index.llms.nvidia import NVIDIA

# connect to an chat NIM running at localhost:8080
llm = NVIDIA(base_url="http://localhost:8080/v1")
```
