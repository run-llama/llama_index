# LlamaIndex Embeddings Integration: NVIDIA NIM Microservices

The `llama-index-embeddings-nvidia` package contains LlamaIndex integrations for building applications with [NVIDIA NIM microservices](https://developer.nvidia.com/nim).
With the NVIDIA embeddings connector, you can connect to, and generate content from, compatible models.

NVIDIA NIM supports models across domains like chat, embedding, and re-ranking, from the community as well as from NVIDIA.
Each model is optimized by NVIDIA to deliver the best performance on NVIDIA-accelerated infrastructure and is packaged as a NIM,
an easy-to-use, prebuilt container that deploys anywhere using a single command on NVIDIA accelerated infrastructure.
At their core, NIM microservices are containers that provide interactive APIs for running inference on an AI Model.

NVIDIA-hosted deployments are available on the [NVIDIA API catalog](https://build.nvidia.com/) to test each NIM.
After you explore, you can download NIM microservices from the API catalog, which is included with the NVIDIA AI Enterprise license.
The ability to run models on-premises or in your own cloud gives your enterprise ownership of your customizations and full control of your IP and AI application.

Use this documentation to learn how to install the `llama-index-embeddings-nvidia` package and use it to connect to a model.
The following example connects to the NVIDIA Retrieval QA E5 Embedding Model.

<!-- Don't link to the model yet because until the reader signs in at a following step, the link might 404 -->

## Install the Package

To install the `llama-index-embeddings-nvidia` package, run the following code.

```bash
pip install llama-index-embeddings-nvidia
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

To submit a query to the [nv-embedqa-e5-v5](https://build.nvidia.com/nvidia/nv-embedqa-e5-v5/modelcard) model,
run the following code.

```python
from llama_index.embeddings.nvidia import NVIDIAEmbedding

embedder = NVIDIAEmbedding(model="nv-embedqa-e5-v5")
embedder.get_query_embedding("What's the weather like in Komchatka?")
```

## Self-host with NVIDIA NIM Microservices

When you are ready to deploy your AI application, you can self-host models with NVIDIA NIM.
For more information, refer to [NVIDIA AI Enterprise](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/).

The following example code connects to a locally-hosted NIM Microservice.

```python
from llama_index.embeddings.nvidia import NVIDIAEmbedding

# connect to an embedding NIM running at localhost:8080
embedder = NVIDIAEmbeddings(base_url="http://localhost:8080/v1")
```

## Related Topics

- [Overview of NeMo Retriever Text Embedding NIM](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/overview.html)
