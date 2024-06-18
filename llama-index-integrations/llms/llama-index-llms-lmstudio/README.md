# LlamaIndex Llms Integration: Lmstudio

```bash
pip install llama-index-llms-lmstudio
```

## Usage Steps

1. Open LM Studio App and go to the Local Server Tab
2. In the Configuration settings, enable Apply Prompt Formatting
3. Load the model of your choice
4. Start your server

```python
from llama_index.llms.lmstudio import LMStudio

llm = LMStudio(
    model_name="Hermes-2-Pro-Llama-3-8B",
    base_url="http://localhost:1234/v1",
    temperature=0.7,
)

messages = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content="You an expert AI assistant. Help User with their queries.",
    ),
    ChatMessage(
        role=MessageRole.USER,
        content="What is the significance of the number 42?",
    ),
]

response = llm.chat(messages=messages)
print(str(response))
```
