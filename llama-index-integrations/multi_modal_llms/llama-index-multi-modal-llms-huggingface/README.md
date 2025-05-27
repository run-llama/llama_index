# LlamaIndex Multi_Modal_Llms Integration: Huggingface

This project integrates Hugging Face's multimodal language models into the LlamaIndex framework, enabling advanced multimodal capabilities for various AI applications.

## Features

- Seamless integration of Hugging Face multimodal models with LlamaIndex
- Support for multiple state-of-the-art vision-language models and their **finetunes**:
  - [Qwen2 Vision](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)
  - [Florence2](https://huggingface.co/collections/microsoft/florence-6669f44df0d87d9c3bfb76de)
  - [Phi-3.5 Vision](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3)
  - [PaLI-Gemma](https://huggingface.co/collections/google/paligemma-release-6643a9ffbf57de2ae0448dda)
- Easy-to-use interface for multimodal tasks like image captioning and visual question answering
- Configurable model parameters for fine-tuned performance

---

## Author of that Integration [GitHub](https://github.com/g-hano) | [LinkedIn](https://www.linkedin.com/in/chanyalcin/) | [Email](mcihan.yalcin@outlook.com)

## Installation

```bash
pip install llama-index-multi-modal-llms-huggingface
```

Make sure to set your Hugging Face API token as an environment variable:

```bash
export HF_TOKEN=your_huggingface_token_here
```

## Usage

Here's a basic example of how to use the Hugging Face multimodal integration:

```python
from llama_index.multi_modal_llms.huggingface import HuggingFaceMultiModal
from llama_index.core.schema import ImageDocument

# Initialize the model
model = HuggingFaceMultiModal.from_model_name("Qwen/Qwen2-VL-2B-Instruct")

# Prepare your image and prompt
image_document = ImageDocument(image_path="path/to/your/image.jpg")
prompt = "Describe this image in detail."

# Generate a response
response = model.complete(prompt, image_documents=[image_document])

print(response.text)
```

### Streaming

```python
from llama_index.multi_modal_llms.huggingface import HuggingFaceMultiModal
from llama_index.core.schema import ImageDocument

# Initialize the model
model = HuggingFaceMultiModal.from_model_name("Qwen/Qwen2-VL-2B-Instruct")

# Prepare your image and prompt
image_document = ImageDocument(image_path="downloaded_image.jpg")
prompt = "Describe this image in detail."

import nest_asyncio
import asyncio

nest_asyncio.apply()


async def stream_output():
    for chunk in model.stream_complete(
        prompt, image_documents=[image_document]
    ):
        print(chunk.delta, end="", flush=True)
        await asyncio.sleep(0)


asyncio.run(stream_output())
```

You can also refer to this [Colab notebook](examples\huggingface_multimodal.ipynb)

## Supported Models

1. Qwen2 Vision
2. Florence2
3. Phi3.5 Vision
4. PaliGemma
5. Mllama

Each model has its unique capabilities and can be selected based on your specific use case.

## Configuration

You can configure various parameters when initializing a model:

```python
model = HuggingFaceMultiModal(
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    device="cuda",  # or "cpu"
    torch_dtype=torch.float16,
    max_new_tokens=100,
    temperature=0.7,
)
```

## Limitations

- Async streaming is not supported for any of the models.
- Some models have specific requirements or limitations. Please refer to the individual model classes for details.

---

## Author of that Integration [GitHub](https://github.com/g-hano) | [LinkedIn](https://www.linkedin.com/in/chanyalcin/) | [Email](mcihan.yalcin@outlook.com)
