# LlamaIndex Embeddings Integration: Huggingface API

Integration with Hugging Face's Inference API for embeddings.

For more information on Hugging Face's Inference API, visit [Hugging Face's Inference API documentation](https://huggingface.co/docs/api-inference/quicktour).

## Installation

```shell
pip install llama-index-embeddings-huggingface-api
```

## Usage

```python
from llama_index.embeddings.huggingface_api import (
    HuggingFaceInferenceAPIEmbedding,
)

my_embed = HuggingFaceInferenceAPIEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    token="<your-token>",  # Optional
)

embeddings = my_embed.get_text_embedding("Why sky is blue")
```
