# LlamaIndex Llms Integration: Openai Like

## Overview

This package is a thin wrapper around the OpenAI API. It is designed to be used with the OpenAI API, but can be used with any OpenAI-compatible API into multimodal language models of the LlamaIndex framework.

## Installation

```bash
pip install llama-index-multi-modal-llms-openai-like
```

## Usage

### Initialization

```python
from llama_index.multi_modal_llms.openai_like import OpenAILikeMultiModal

llm = OpenAILikeMultiModal(
    model="Qwen/Qwen2-VL-72B-Instruct-base",
    api_base="http://localhost:1234/v1",
)
```

#### Without environmental variables

```python
from llama_index.multi_modal_llms.openai_like import OpenAILikeMultiModal

llm = OpenAILikeMultiModal(
    api_key="your_api_key",
    model="Qwen/Qwen2-VL-72B-Instruct-base",
    api_base="http://localhost:1234/v1",
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
