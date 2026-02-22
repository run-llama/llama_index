# LlamaIndex LLMs Integration: HuggingFace-LangChain

This package provides a LlamaIndex LLM integration using [langchain-huggingface](https://python.langchain.com/docs/integrations/platforms/huggingface/), combining the convenience of LangChain's HuggingFace wrappers with LlamaIndex's query and indexing capabilities.

## Installation

```bash
pip install llama-index-llms-huggingface-langchain
```

## Usage

### Remote via HuggingFace Inference API

```python
from llama_index.llms.huggingface_langchain import HuggingFaceLangChainLLM

llm = HuggingFaceLangChainLLM(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    max_new_tokens=512,
    temperature=0.1,
)

# Text completion
response = llm.complete("Explain quantum computing in simple terms.")
print(response.text)

# Chat
from llama_index.core.base.llms.types import ChatMessage, MessageRole

messages = [
    ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
    ChatMessage(role=MessageRole.USER, content="What is Python?"),
]
chat_response = llm.chat(messages)
print(chat_response.message.content)
```

### Local execution

```python
llm = HuggingFaceLangChainLLM(
    repo_id="google/flan-t5-small",
    backend="local",
    task="text2text-generation",
)

response = llm.complete("Translate to French: Hello, how are you?")
print(response.text)
```

### Parameters

- `repo_id` (str): HuggingFace model repository ID
- `task` (str): Model task type (default: `"text-generation"`)
- `backend` (str): `"api"` for HuggingFace Inference API, `"local"` for local transformers pipeline
- `max_new_tokens` (int): Maximum tokens to generate (default: 256)
- `temperature` (float): Sampling temperature (default: 0.1)
- `do_sample` (bool): Use sampling vs greedy decoding (default: False)
- `is_chat_model` (bool): Wrap with ChatHuggingFace for chat templates (default: True)
- `huggingfacehub_api_token` (str): HuggingFace API token (falls back to `HF_TOKEN` env var)
- `model_kwargs` (dict): Additional model constructor arguments

## Why this package?

This package bridges **langchain-huggingface** (which provides `HuggingFaceEndpoint`, `ChatHuggingFace`, and `HuggingFacePipeline`) with **LlamaIndex**, giving you:

- Proper chat template handling via `ChatHuggingFace`
- Access to HuggingFace's Inference API with automatic task routing
- Local model execution via transformers pipelines
- Full LlamaIndex compatibility (indexing, querying, RAG pipelines)
