# LlamaIndex Llms Integration: ModelScope

## Installation

To install the required package, run:

```bash
!pip install llama-index-llms-modelscope
```

## Basic Usage

### Initialize the ModelScopeLLM

To use the ModelScopeLLM model, create an instance by specifying the model name and revision:

```python
import sys
from llama_index.llms.modelscope import ModelScopeLLM

llm = ModelScopeLLM(model_name="qwen/Qwen3-8B", model_revision="master")
```

### Generate Completions

To generate a text completion for a prompt, use the `complete` method:

```python
rsp = llm.complete("Hello, who are you?")
print(rsp)
```

### Using Message Requests

You can chat with the model by using a list of messages. Here’s how to set it up:

```python
from llama_index.core.base.llms.types import MessageRole, ChatMessage

messages = [
    ChatMessage(
        role=MessageRole.SYSTEM, content="You are a helpful assistant."
    ),
    ChatMessage(role=MessageRole.USER, content="How to make cake?"),
]
resp = llm.chat(messages)
print(resp)
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/modelscope/
