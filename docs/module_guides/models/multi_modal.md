# [Beta] Multi-modal models

## Concept

Large language models (LLMs) are text-in, text-out. Large Multi-modal Models (LMMs) generalize this beyond the text modalities. For instance, models such as GPT-4V allow you to jointly input both images and text, and output text.

We've included a base `MultiModalLLM` abstraction to allow for text+image models. **NOTE**: This naming is subject to change!

## Usage Pattern

The following code snippet shows how you can get started using LMMs e.g. with GPT-4V.

```python
from llama_index.multi_modal_llms import OpenAIMultiModal
from llama_index.multi_modal_llms.generic_utils import (
    load_image_urls,
)
from llama_index import SimpleDirectoryReader

# load image documents from urls
image_documents = load_image_urls(image_urls)

# load image documents from local directory
image_documents = SimpleDirectoryReader(local_directory).load_data()

# non-streaming
openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview", api_key=OPENAI_API_TOKEN, max_new_tokens=300
)
response = openai_mm_llm.complete(
    prompt="what is in the image?", image_documents=image_documents
)
```

## Modules

We support integrations with GPT-4V, LLaVA, and more.

```{toctree}
---
maxdepth: 1
---
/examples/multi_modal/openai_multi_modal.ipynb
/examples/multi_modal/replicate_multi_modal.ipynb
/examples/multi_modal/multi_modal_retrieval.ipynb
/examples/multi_modal/llava_multi_modal_tesla_10q.ipynb
```
