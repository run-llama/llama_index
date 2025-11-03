# LlamaIndex Postprocessor Integration: NVIDIA NIM Microservices

The `llama-index-postprocessor-nvidia-rerank` package contains LlamaIndex integrations for building applications with [NVIDIA NIM microservices](https://developer.nvidia.com/nim).
With the NVIDIA postprocessor connector, you can use a reranking NIM to rerank processed data.

NVIDIA NIM supports models across domains like chat, embedding, and re-ranking, from the community as well as from NVIDIA.
Each model is optimized by NVIDIA to deliver the best performance on NVIDIA-accelerated infrastructure and is packaged as a NIM,
an easy-to-use, prebuilt container that deploys anywhere using a single command on NVIDIA accelerated infrastructure.
At their core, NIM microservices are containers that provide interactive APIs for running inference on an AI Model.

NVIDIA-hosted deployments are available on the [NVIDIA API catalog](https://build.nvidia.com/) to test each NIM.
After you explore, you can download NIM microservices from the API catalog, which is included with the NVIDIA AI Enterprise license.
The ability to run models on-premises or in your own cloud gives your enterprise ownership of your customizations and full control of your IP and AI application.

Use this documentation to learn how to install the `llama-index-postprocessor-nvidia-rerank` package
and use it to rerank parsed nodes.

## Install the Package

To install the `llama-index-postprocessor-nvidia-rerank` package, run the following code.

```shell
pip install --upgrade llama-index llama-index-core llama-index-postprocessor-nvidia-rerank
```

## Access the NVIDIA API Catalog

To get access to the NVIDIA API Catalog, do the following:

1. Create a free account on the [NVIDIA API Catalog](https://build.nvidia.com/) and log in.
2. Click your profile icon, and then click **API Keys**. The **API Keys** page appears.
3. Click **Generate API Key**. The **Generate API Key** window appears.
4. Click **Generate Key**. You should see **API Key Granted**, and your key appears.
5. Copy and save the key as `NVIDIA_API_KEY`.
6. To verify your key, use the following code.

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

You can now use your key to access endpoints on the NVIDIA API Catalog.

## Work with the API Catalog

The following example loads and parses data, and then calls the reranking NIM.

```python
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank

from llama_index.core import Document
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter, SimpleFileNodeParser

# Load your API key from an environment variable
my_key = os.environ["NVIDIA_API_KEY"]

# Load documents
documents = SimpleDirectoryReader("/path_to_your_data_folder").load_data()

# Set the reranker to use the API Catalog's default reranker model
rerank = NVIDIARerank()

# Parse data into nodes
parser = SentenceSplitter(separator="\n", chunk_size=200, chunk_overlap=0)
nodes = parser.get_nodes_from_documents(documents)

# Rerank the nodes
rerank.postprocess_nodes(nodes, query_str="What is the capital of France?")
```

## Available Models

You can querying `available_models` to get a list of the models available with your API credentials.
For details about each model, refer to [Models](https://docs.api.nvidia.com/nim/reference/models-1).

```python
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank

rerank.available_models
```

## Self-host with NVIDIA NIM Microservices

When you are ready to deploy your AI application, you can self-host models with NVIDIA NIM.
For more information, refer to [NVIDIA AI Enterprise](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/).

The following example code connects to a locally-hosted NIM Microservice.

```python
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank

# connect to a reranking NIM running at localhost:1976
rerank = NVIDIARerank(base_url="http://localhost:1976/v1")
```

## Use Your Own Custom HTTP Client

If you need more control over HTTP settings, such as timeouts, proxies, and retries, you can use your own custom HTTP client.
Use the following code to pass an instance of your HTTP client to the `NVIDIARerank` initializer.

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

## Related Topics

- [Overview of NeMo Retriever Text Reranking NIM](https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/overview.html)
