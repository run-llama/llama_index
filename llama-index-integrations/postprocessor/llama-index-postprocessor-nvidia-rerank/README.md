# LlamaIndex Postprocessor Integration: Nvidia_Rerank

The `llama-index-postprocessor-nvidia-rerank` package contains LlamaIndex integrations for rerank model powered by the [NVIDIA AI Foundation Model](https://www.nvidia.com/en-us/ai-data-science/foundation-models/) playground environment.

> [NVIDIA AI Foundation Endpoints](https://www.nvidia.com/en-us/ai-data-science/foundation-models/) give users easy access to hosted endpoints for generative AI models like Llama-2, SteerLM, Mistral, etc. Using the API, you can query live endpoints available on the [NVIDIA GPU Cloud (NGC)](https://catalog.ngc.nvidia.com/ai-foundation-models) to get quick results from a DGX-hosted cloud compute environment. All models are source-accessible and can be deployed on your own compute cluster.
> For more information about documentation, please visit [NVIDIA NeMo Retriever Reranking](https://developer.nvidia.com/docs/nemo-microservices/reranking/source/overview.html).

Below is an example on how to use some common functionality surrounding text-generative and embedding models

## Installation

```shell
pip install --upgrade llama-index llama-index-core llama-index-nvidia-rerank
```

## Setup

**To get started:**

1. Create a free account with the [NVIDIA GPU Cloud](https://catalog.ngc.nvidia.com/) service, which hosts AI solution catalogs, containers, models, etc.
2. Navigate to `Catalog > AI Foundation Models > (Model with API endpoint)`.
3. Select the `API` option and click `Generate Key`.
4. Save the generated key as `NVIDIA_API_KEY`. From there, you should have access to the endpoints.

This is how you set NVIDIA_API_KEY in environment variable
export NVIDIA_API_KEY="Your_NVIDIA_API_KEY_obtained_from_above_setup"

```python
import getpass
import os

if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    nvidia_api_key = getpass.getpass("Enter your NVIDIA AIPLAY API key: ")
    assert nvidia_api_key.startswith(
        "nvapi-"
    ), f"{nvidia_api_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvidia_api_key
```

```python
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank

# for local hosted nim exposed api end point
rerank = NVIDIARerank().mode(
    mode="nim", base_url="http://<your_end_point>:1976/v1"
)
# for API Catalog reranker model
my_key = os.environ["NVIDIA_API_KEY"]
rerank = NVIDIARerank().mode(mode="nvidia", api_key=my_key)
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
import os

# load documents
documents = SimpleDirectoryReader("/path_to_your_data_folder").load_data()

# use API Catalog's reranker model
my_key = os.environ["NVIDIA_API_KEY"]
rerank = NVIDIARerank().mode(mode="nvidia", api_key=my_key)

# parse nodes
parser = SentenceSplitter(separator="\n", chunk_size=200, chunk_overlap=0)
nodes = parser.get_nodes_from_documents(documents)
# rerank
rerank.postprocess_nodes(nodes, query_str=query)
```
