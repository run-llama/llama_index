# LlamaIndex Multi_Modal Integration: Nvidia

This project integrates Nvidia vlm into the LlamaIndex framework, enabling advanced multimodal capabilities for various AI applications.

## Features

- Seamless integration of NVIDIA vlm with LlamaIndex
- Support for multiple state-of-the-art vision-language models:
  - [adept/fuyu-8b](https://build.nvidia.com/adept/fuyu-8b)
  - [google/deplot](https://build.nvidia.com/google/google-deplot)
  - [nvidia/neva-22b](https://build.nvidia.com/nvidia/neva-22b)
  - [google/paligemma](https://build.nvidia.com/google/google-paligemma)
  - [microsoft/phi-3-vision-128k-instruct](https://build.nvidia.com/microsoft/phi-3-vision-128k-instruct)
  - [microsoft/phi-3.5-vision-instruct](https://build.nvidia.com/microsoft/phi-3_5-vision-instruct)
  - [nvidia/vila](https://build.nvidia.com/nvidia)
  - [meta/llama-3.2-11b-vision-instruct](https://build.nvidia.com/meta/llama-3.2-11b-vision-instruct)
  - [meta/llama-3.2-90b-vision-instruct](https://build.nvidia.com/meta/llama-3.2-90b-vision-instruct)
- Easy-to-use interface for multimodal tasks like image captioning and visual question answering
- Configurable model parameters for fine-tuned performance

---

## Installation

```bash
pip install llama-index-multi-modal-llms-nvidia
```

Make sure to set your NVIDIA API key as an environment variable:

```bash
export NVIDIA_API_KEY=your_api_key_here
```

## Usage

Here's a basic example of how to use the Nvidia vlm integration:

```python
from llama_index.multi_modal_llms.nvidia import NVIDIAMultiModal
from llama_index.core.schema import ImageDocument

# Initialize the model
model = NVIDIAMultiModal()

# Prepare your image and prompt
image_document = ImageDocument(image_path="path/to/your/image.jpg")
prompt = "Describe this image in detail."

# Generate a response
response = model.complete(prompt, image_documents=[image_document])

print(response.text)
```

### Streaming

```python
from llama_index.multi_modal_llms.nvidia import NVIDIAMultiModal
from llama_index.core.schema import ImageDocument

# Initialize the model
model = NVIDIAMultiModal()

# Prepare your image and prompt
image_document = ImageDocument(image_path="downloaded_image.jpg")
prompt = "Describe this image in detail."

import nest_asyncio
import asyncio

nest_asyncio.apply()

response = model.stream_complete(
    prompt=f"Describe the image",
    image_documents=[
        ImageDocument(metadata={"asset_id": asset_id}, mimetype="png")
    ],
)

for r in response:
    print(r.text, end="")
```

## Passing an image as an NVCF asset

If your image is sufficiently large or you will pass it multiple times in a chat conversation, you may upload it once and reference it in your chat conversation

See https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/assets.html for details about how upload the image.

```python
import requests

content_type = "image/jpg"
description = "example-image-from-lc-nv-ai-e-notebook"

create_response = requests.post(
    "https://api.nvcf.nvidia.com/v2/nvcf/assets",
    headers={
        "Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}",
        "accept": "application/json",
        "Content-Type": "application/json",
    },
    json={"contentType": content_type, "description": description},
)
create_response.raise_for_status()

upload_response = requests.put(
    create_response.json()["uploadUrl"],
    headers={
        "Content-Type": content_type,
        "x-amz-meta-nvcf-asset-description": description,
    },
    data=img_response.content,
)
upload_response.raise_for_status()

asset_id = create_response.json()["assetId"]

response = llm.complete(
    prompt=f"Describe the image",
    image_documents=[
        ImageDocument(metadata={"asset_id": asset_id}, mimetype="png")
    ],
)
```
