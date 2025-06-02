# LlamaIndex Llms Integration: Aliyun-PaiEas

The `llama-index-llms-paieas` package contains LlamaIndex integrations building applications with models on
[Aliyun PAI-EAS(Elastic Algorithm Service)](https://www.alibabacloud.com/help/en/pai/user-guide/purchase-and-configure-eas-resource-groups/) ([Chinese Version](https://help.aliyun.com/zh/pai/user-guide/overview-2?spm=5176.28273668.J_3928357510.5.4ec51c6fowCIrH))LLM inference service.

The Elastic Algorithm Service (EAS) is a model online service platform that allows you to deploy models as online reasoning services or AI-Web applications with one click. It provides functions such as elastic scaling and blue-green deployment, which can support you to obtain high-concurrency and stable online algorithm model services at a lower resource cost. In addition, EAS also provides functions such as resource group management, version control, and resource monitoring, which facilitates you to apply model services to your business. EAS is suitable for a variety of AI reasoning scenarios such as real-time synchronous reasoning and near-real-time asynchronous reasoning, and has the capabilities of a complete operation and maintenance monitoring system.

# PAI EAS's LLM Service

This example goes over how to use LlamaIndex to interact with and develop LLM-powered systems using the PAI-EAS LLM service endpoints.

With this endpoint, you'll be able to connect to various open-source LLM, such as:

- Tongyi's [qwen](https://huggingface.co/Qwen)
- Google's [gemma-7b](https://build.nvidia.com/google/gemma-7b)
- Mistal AI's [mistral-7b-instruct-v0.2](https://build.nvidia.com/mistralai/mistral-7b-instruct-v2)
- And more!

## Installation

```shell
pip install llama-index-llms-paieas
```

## Setup

**To get started:**

1. Create a free account with [Aliyun PAI EAS](https://common-buy.aliyun.com/?spm=5176.28273668.J_3928357510.3.4ec51c6fowCIrH&commodityCode=learn_EasDedicatedPostpay_public_cn) and login.

2. Refer the guide(https://help.aliyun.com/zh/pai/use-cases/deploy-llm-in-eas?spm=a2c4g.11186623.0.0.63de3c57Ef3r3C#f3ef8927ec4l2), on the EAS page, click Deploy LLM Service.

3. Under the Basic Information, on the Public Address Call tab, click View Call Information, and get the service token `PAIEAS_API_KEY` and access address `PAIEAS_API_BASE`.

4. Copy and save the service token as `PAIEAS_API_KEY` and access address as `PAIEAS_API_BASE`.

```python
import os

os.environ["PAIEAS_API_KEY"] = your_service_token
os.environ["PAIEAS_API_BASE"] = your_access_address
```

## Working with API Catalog

```python
from llama_index.llms.paieas import PaiEas
from llama_index.core.llms import ChatMessage, MessageRole

llm = PaiEas()

messages = [
    ChatMessage(
        role=MessageRole.SYSTEM, content=("You are a helpful assistant.")
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=("What is Alibaba Cloud?"),
    ),
]

llm.chat(messages)
```
