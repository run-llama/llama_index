# LlamaIndex Llms Integration: Text Generation Inference

Integration with [Text Generation Inference](https://huggingface.co/docs/text-generation-inference) from Hugging Face to generate text.

## Installation

```shell
pip install llama-index-llms-text-generation-inference
```

## Usage

```python
from llama_index.llms.text_generation_inference import TextGenerationInference

llm = TextGenerationInference(
    model_name="openai-community/gpt2",
    temperature=0.7,
    max_tokens=100,
    token="<your-token>",  # Optional
)

response = llm.complete("Hello, how are you?")
```
