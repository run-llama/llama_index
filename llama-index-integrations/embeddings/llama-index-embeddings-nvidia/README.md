# NVIDIA NIMs

The `llama-index-embeddings-nvidia` package contains LlamaIndex integration for embeddings powered by [NVIDIA AI Foundation Models](https://www.nvidia.com/en-us/ai-data-science/foundation-models/) and hosted on [NVIDIA API Catalog.](https://build.nvidia.com/)

NVIDIA AI Foundation models are community and NVIDIA-built models and are NVIDIA-optimized to deliver the best performance on NVIDIA accelerated infrastructure.  Using the API, you can query live endpoints available on the NVIDIA API Catalog to get quick results from a DGX-hosted cloud compute environment. All models are source-accessible and can be deployed on your own compute cluster using NVIDIA NIM which is part of NVIDIA AI Enterprise.

Models can be exported from NVIDIA’s API catalog with NVIDIA NIM, which is included with the NVIDIA AI Enterprise license, and run them on-premises, giving Enterprises ownership of their customizations and full control of their IP and AI application. NIMs are packaged as container images on a per model/model family basis and are distributed as NGC container images through the NVIDIA NGC Catalog. At their core, NIMs are containers that provide interactive APIs for running inference on an AI Model. 

# NVIDIA's Embeddings connector

With this connector, you'll be able to connect to and generate from compatible models available and hosted on [NVIDIA API Catalog](https://build.nvidia.com/), such as:

- NVIDIA's Retrieval QA Embedding Model [embed-qa-4](https://build.nvidia.com/nvidia/embed-qa-4)

## Installation

```bash
pip install llama-index-embeddings-nvidia
```
## Setup

**To get started:**

1. Create a free account with [NVIDIA](https://build.nvidia.com/), which hosts NVIDIA AI Foundation models.

2. Select the `Retrieval` tab, then select your model of choice.

3. Under `Input` select the `Python` tab, and click `Get API Key`. Then click `Generate Key`.

4. Copy and save the generated key as `NVIDIA_API_KEY`. From there, you should have access to the endpoints.

```python
import getpass
import os

if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    print("Valid NVIDIA_API_KEY already in environment. Delete to reset")
else:
    nvapi_key = getpass.getpass("NVAPI Key (starts with nvapi-): ")
    assert nvapi_key.startswith("nvapi-"), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvapi_key
```

## Working with API Catalog

```python
from llama_index.embeddings.nvidia import NVIDIAEmbedding

embedder = NVIDIAEmbedding()
embedder.get_query_embedding("What's the weather like in Komchatka?")
```

## Working with NVIDIA NIMs
When ready to deploy, you can self-host models with NVIDIA NIM—which is included with the NVIDIA AI Enterprise software license—and run them anywhere, giving you ownership of your customizations and full control of your intellectual property (IP) and AI applications.

[Learn more about NIMs](https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/)

```python
from llama_index.embeddings.nvidia import NVIDIAEmbedding

# connect to an embedding NIM running at localhost:2016
embedder = NVIDIAEmbeddings(base_url="http://localhost:2016/v1")
```

