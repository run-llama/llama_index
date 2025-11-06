# LlamaIndex LLMs Integration: NVIDIA NIM for LLMs

The `llama-index-llms-nvidia` package contains LlamaIndex integrations for building applications with [NVIDIA NIM](https://developer.nvidia.com/nim).
With the NVIDIA LLM connector, you can develop LLM-powered systems using [NVIDIA AI Foundation models](https://www.nvidia.com/en-us/ai-data-science/foundation-models/).

NVIDIA NIM for LLM supports models across domains like chat, reward, and reasoning, from the community as well as from NVIDIA.
Each model is optimized by NVIDIA to deliver the best performance on NVIDIA-accelerated infrastructure and is packaged as a NIM,
an easy-to-use, prebuilt container that deploys anywhere using a single command on NVIDIA accelerated infrastructure.
At their core, NIM for LLMs are containers that provide interactive APIs for running inference on an AI Model.

NVIDIA-hosted deployments are available on the [NVIDIA API catalog](https://build.nvidia.com/) to test each NIM.
After you explore, you can download NIM for LLMs from the API catalog, which is included with the NVIDIA AI Enterprise license.
The ability to run models on-premises or in your own cloud gives your enterprise ownership of your customizations and full control of your IP and AI application.

Use this documentation to learn how to install the `llama-index-llms-nvidia` package
and use it to connect to, and generate content from, compatible LLM models.

## Install the Package

To install the `llama-index-llms-nvidia` package, run the following code.

```shell
pip install llama-index-llms-nvidia
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

The following example chats with the default LLM.

```python
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.llms import ChatMessage, MessageRole

# Use the default model
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

For models that are not included in the [CHAT_MODEL_TABLE](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/llms/llama-index-llms-nvidia/llama_index/llms/nvidia/utils.py), you must explicitly specify whether the model supports chat endpoints.
Set the `is_chat_model` parameter as described following:

- `False` – Use the `/completions` endpoint. This is the default value.
- `True` – Use the `/chat/completions` endpoint.

The following example chats with the Llama-3.3-Nemotron-Super-49B-v1 LLM.

```python
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.llms import ChatMessage, MessageRole

# Use a specific model
llm = NVIDIA(
    model="nvidia/llama-3.3-nemotron-super-49b-v1", is_chat_model=True
)

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

## Self-host with NVIDIA NIM for LLMs

When you are ready to deploy your AI application, you can self-host models with NVIDIA NIM for LLMs.
For more information, refer to [NVIDIA AI Enterprise](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/).

The following example code connects to a locally-hosted LLM.

```python
from llama_index.llms.nvidia import NVIDIA

# connect to an chat NIM running at localhost:8080
llm = NVIDIA(base_url="http://localhost:8080/v1")
```

## Related Topics

- [Overview of NVIDIA NIM for Large Language Models (LLMs)](https://docs.nvidia.com/nim/large-language-models/latest/introduction.html)
