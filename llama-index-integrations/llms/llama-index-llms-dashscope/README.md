# LlamaIndex Llms Integration: Dashscope

## Installation

1. Install the required Python package:

   ```bash
   pip install llama-index-llms-dashscope
   ```

2. Set the DashScope API key as an environment variable:

   ```bash
   export DASHSCOPE_API_KEY=YOUR_DASHSCOPE_API_KEY
   ```

   Alternatively, you can set it in your Python script:

   ```python
   import os

   os.environ["DASHSCOPE_API_KEY"] = "YOUR_DASHSCOPE_API_KEY"
   ```

## Usage

### Basic Recipe Generation

To generate a basic vanilla cake recipe:

```python
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels

# Initialize DashScope object
dashscope_llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX)

# Generate a vanilla cake recipe
resp = dashscope_llm.complete("How to make cake?")
print(resp)
```

### Streaming Recipe Responses

For real-time streamed responses:

```python
responses = dashscope_llm.stream_complete("How to make cake?")
for response in responses:
    print(response.delta, end="")
```

### Multi-Round Conversation

To have a conversation with the assistant and ask for a sugar-free cake recipe:

```python
from llama_index.core.base.llms.types import MessageRole, ChatMessage

messages = [
    ChatMessage(
        role=MessageRole.SYSTEM, content="You are a helpful assistant."
    ),
    ChatMessage(role=MessageRole.USER, content="How to make cake?"),
]

# Get first round response
resp = dashscope_llm.chat(messages)
print(resp)

# Continue conversation
messages.append(
    ChatMessage(role=MessageRole.ASSISTANT, content=resp.message.content)
)
messages.append(
    ChatMessage(role=MessageRole.USER, content="How to make it without sugar?")
)

# Get second round response
resp = dashscope_llm.chat(messages)
print(resp)
```

### Handling Sugar-Free Recipes

For sugar-free cake recipes using honey as a sweetener:

```python
resp = dashscope_llm.complete("How to make cake without sugar?")
print(resp)
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/dashscope/
