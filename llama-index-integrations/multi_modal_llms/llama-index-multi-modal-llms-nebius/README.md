# LlamaIndex Llms Integration: [Nebius AI Studio](https://studio.nebius.ai/)

## Overview

This project integrates Nebius AI Studio API into multimodal language models of the LlamaIndex framework.

## Installation

```bash
pip install llama-index-multi-modal-llms-nebius
```

## Usage

### Initialization

#### With environmental variables.

```.env
NEBIUS_API_KEY=your_api_key

```

```python
from llama_index.multi_modal_llms.nebius import NebiusMultiModal

llm = NebiusLLM(model="Qwen/Qwen2-VL-72B-Instruct-base")
```

#### Without environmental variables

```python
from llama_index.multi_modal_llms.nebius import NebiusMultiModal

llm = NebiusLLM(
    api_key="your_api_key", model="Qwen/Qwen2-VL-72B-Instruct-base"
)
```

### Launching

#### Load images

```python
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls

image_urls = [
    "https://res.cloudinary.com/hello-tickets/image/upload/c_limit,f_auto,q_auto,w_1920/v1640835927/o3pfl41q7m5bj8jardk0.jpg",
]
image_documents = load_image_urls(image_urls)
```

#### Call `complete` with a prompt

```python
complete_response = mm_llm.complete(
    prompt="Describe the images as an alternative text",
    image_documents=image_documents,
)
```

#### Stream `complete`

```python
stream_complete_response = mm_llm.stream_complete(
    prompt="give me more context for this image",
    image_documents=image_documents,
)
for r in stream_complete_response:
    print(r.delta, end="")
```
