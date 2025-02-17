# NVIDIA NIMs

The `llama-index-postprocessor-nvidia-rerank` package contains LlamaIndex integrations building applications with models on
NVIDIA NIM inference microservice. NIM supports models across domains like chat, embedding, and re-ranking models
from the community as well as NVIDIA. These models are optimized by NVIDIA to deliver the best performance on NVIDIA
accelerated infrastructure and deployed as a NIM, an easy-to-use, prebuilt containers that deploy anywhere using a single
command on NVIDIA accelerated infrastructure.

NVIDIA hosted deployments of NIMs are available to test on the [NVIDIA API catalog](https://build.nvidia.com/). After testing,
NIMs can be exported from NVIDIA’s API catalog using the NVIDIA AI Enterprise license and run on-premises or in the cloud,
giving enterprises ownership and full control of their IP and AI application.

NIMs are packaged as container images on a per model basis and are distributed as NGC container images through the NVIDIA NGC Catalog.
At their core, NIMs provide easy, consistent, and familiar APIs for running inference on an AI model.

# LlamaIndex Postprocessor Integration: Nvidia_Rerank

Below is an example on how to use some common functionality surrounding text-generative and embedding models

## Installation

```shell
pip install --upgrade llama-index llama-index-core llama-index-postprocessor-nvidia-rerank
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
    assert nvapi_key.startswith(
        "nvapi-"
    ), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvapi_key
```

## Working with NVIDIA API Catalog

```python
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank

rerank = NVIDIARerank()
```

## Working with NVIDIA NIMs

```python
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank

# connect to an reranking NIM running at localhost:1976
rerank = NVIDIARerank(base_url="http://localhost:1976/v1")
```

## Supported models

Querying `available_models` will still give you all of the other models offered by your API credentials.

```python
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank

rerank.available_models
```

**To find out more about a specific model, please navigate to the NVIDIA NIM section of ai.nvidia.com [as linked here](https://docs.api.nvidia.com/nim/).**

## Reranking

Below is an example:

```python
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank

from llama_index.core import Document
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter, SimpleFileNodeParser


# load documents
documents = SimpleDirectoryReader("/path_to_your_data_folder").load_data()

# use API Catalog's reranker model
my_key = os.environ["NVIDIA_API_KEY"]
rerank = NVIDIARerank()

# parse nodes
parser = SentenceSplitter(separator="\n", chunk_size=200, chunk_overlap=0)
nodes = parser.get_nodes_from_documents(documents)
# rerank
rerank.postprocess_nodes(nodes, query_str=query)
```

### Custom HTTP Client

If you need more control over HTTP settings (e.g., timeouts, proxies, retries), you can pass your own `httpx.Client` instance to the `NVIDIARerank` initializer:

```python
import httpx
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank

# Create a custom httpx client with a 10-second timeout
custom_client = httpx.Client(timeout=10.0)

# Pass the custom client to the reranker
rerank = NVIDIARerank(
    base_url="http://localhost:1976/v1", http_client=custom_client
)
```
