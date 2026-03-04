# LlamaIndex Llms Integration: Text Generation Inference

⚠️ This integration has been deprecated!

The `TextGenerationInference` is no longer maintained. Instead, you can use [`HuggingFaceInferenceAPI`](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/llms/llama-index-llms-huggingface-api). The underlying Text Generation Inference SDK (`tgi`) [has been deprecated](https://github.com/huggingface/text-generation-inference/tree/main/clients/python) in favor of `huggingface_hub`, which `HuggingFaceInferenceAPI` is built on top of.

Instead, use `llama-index-llms-huggingface-api`:

```shell
pip install llama-index-llms-huggingface-api
```

Usage:

```py
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

# access hugging face inference
hub_llm = HuggingFaceInferenceAPI(model="openai-community/gpt2")
# or with a local TGI server
tgi_llm = HuggingFaceInferenceAPI(model="http://localhost:8080")
```
