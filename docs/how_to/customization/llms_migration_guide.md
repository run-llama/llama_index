# Migration Guide for Using LLMs in LlamaIndex

We have made some changes to the configuration of LLMs in LLamaIndex to improve its functionality and ease of use.

Previously, the primary abstraction for an LLM was the `LLMPredictor`. However, we have upgraded to a new abstraction called `LLM`, which offers a cleaner and more user-friendly interface.

These changes will only affect you if you were using the `ChatGPTLLMPredictor`, `HuggingFaceLLMPredictor`, or a custom implementation subclassing `LLMPredictor`.

## If you were using `ChatGPTLLMPredictor`:
We have removed the `ChatGPTLLMPredictor`, but you can still achieve the same functionality using our new `OpenAI` class.

## If you were using `HuggingFaceLLMPredictor`:
We have updated the Hugging Face support to utilize the latest `LLM` abstraction through `HuggingFaceLLM`. To use it, initialize the `HuggingFaceLLM` in the same way as before. Instead of passing it as the `llm_predictor` argument to the service context, you now need to pass it as the `llm` argument.

Old:
```python
hf_predictor = HuggingFaceLLMPredictor(...)
service_context = ServiceContext.from_defaults(llm_predictor=hf_predictor)
```

New:
```python
llm = HuggingFaceLLM(...)
service_context = ServiceContext.from_defaults(llm=llm)
```

## If you were subclassing `LLMPredictor`:
We have refactored the `LLMPredictor` class and removed some outdated logic, which may impact your custom class. The recommended approach now is to implement the `llama_index.llms.base.LLM` interface when defining a custom LLM. Alternatively, you can subclass the simpler `llama_index.llms.custom.CustomLLM` interface.

Here's an example:

```python
from llama_index.llms.base import CompletionResponse, LLMMetadata, StreamCompletionResponse
from llama_index.llms.custom import CustomLLM

class YourLLM(CustomLLM):
    def __init__(self, ...): 
        # initialization logic
        pass

    @property
    def metadata(self) -> LLMMetadata:
        # metadata
        pass

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # completion endpoint
        pass

    def stream_complete(self, prompt: str, **kwargs: Any) -> StreamCompletionResponse:
        # streaming completion endpoint
        pass
```

For further reference, you can look at `llama_index/llms/huggingface.py`.