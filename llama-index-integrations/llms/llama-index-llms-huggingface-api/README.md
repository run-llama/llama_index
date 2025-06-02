# LlamaIndex Llms Integration: Huggingface API

Integration with Hugging Face's Inference API for generating text.

For more information on Hugging Face's Inference API, visit [Hugging Face's Inference API documentation](https://huggingface.co/docs/api-inference/quicktour).

## Installation

```shell
pip install llama-index-llms-huggingface-api
```

## Usage

```python
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

llm = HuggingFaceInferenceAPI(
    model_name="openai-community/gpt2",
    temperature=0.7,
    max_tokens=100,
    token="<your-token>",  # Optional
    provider="hf-inference",  # Optional
)

response = llm.complete("Hello, how are you?")
```
