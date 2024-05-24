# NVIDIA NIMs

The `llama-index-postprocessor-nvidia-rerank` package contains LlamaIndex integrations for rerank model powered by the [NVIDIA AI Foundation Model](https://www.nvidia.com/en-us/ai-data-science/foundation-models/) and hosted on [NVIDIA API Catalog.](https://build.nvidia.com/)

NVIDIA AI Foundation models are community and NVIDIA-built models and are NVIDIA-optimized to deliver the best performance on NVIDIA accelerated infrastructure.  Using the API, you can query live endpoints available on the NVIDIA API Catalog to get quick results from a DGX-hosted cloud compute environment. All models are source-accessible and can be deployed on your own compute cluster using NVIDIA NIM which is part of NVIDIA AI Enterprise.

Models can be exported from NVIDIA’s API catalog with NVIDIA NIM, which is included with the NVIDIA AI Enterprise license, and run them on-premises, giving Enterprises ownership of their customizations and full control of their IP and AI application. NIMs are packaged as container images on a per model/model family basis and are distributed as NGC container images through the NVIDIA NGC Catalog. At their core, NIMs are containers that provide interactive APIs for running inference on an AI Model.

# LlamaIndex Postprocessor Integration: Nvidia_Rerank

Below is an example on how to use some common functionality surrounding text-generative and embedding models

## Installation

```shell
pip install --upgrade llama-index llama-index-core llama-index-nvidia-rerank
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

## Working with NVIDIA API Catalog
```python
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank

rerank = NVIDIARerank()
```

## Working with NVIDIA NIMs

```python
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank

# connect to an embedding NIM running at localhost:2016
rerank = NVIDIARerank(base_url="http://localhost:2016/v1")
```

## Supported models

Querying `get_available_models` will still give you all of the other models offered by your API credentials.

```python
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank

NVIDIARerank.get_available_models()
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
