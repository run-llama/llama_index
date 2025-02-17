# LlamaIndex Llms Integration: SiliconFlow

## 1. Product Introduction

SiliconCloud provides cost-effective GenAI services based on an excellent open-source foundation model.
introduction: https://docs.siliconflow.cn/introduction

## 2. Product features

- As a one-stop cloud service platform that integrates top large models, SiliconCloud is committed to providing developers with faster, cheaper, more comprehensive, and smoother model APIs.

  - SiliconCloud has been listed on Qwen2.5-72B, DeepSeek-V2.5, Qwen2, InternLM2.5-20B-Chat, BCE, BGE, SenseVoice-Small, Llama-3.1, FLUX.1, DeepSeek-Coder-V2, SD3 Medium, GLM-4-9B-Chat, A variety of open-source large language models, image generation models, code generation models, vector and reordering models, and multimodal large models, including InstantID.

  - Among them, Qwen 2.5 (7B), Llama 3.1 (8B) and other large model APIs are free to use, so that developers and product managers do not need to worry about the computing power costs caused by the R&D stage and large-scale promotion, and realize "token freedom".

- Provide out-of-the-box large model inference acceleration services to bring a more efficient user experience to your GenAI applications.

## 3. Installation

```shell
pip install llama-index-llms-siliconflow
```

## 4. Usage

### Complete/Chat

```python
import asyncio
import os
from llama_index.core.llms import ChatMessage
from llama_index.llms.siliconflow import SiliconFlow

llm = SiliconFlow(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
)

response = llm.complete("...")
print(response)

response = asyncio.run(llm.acomplete("..."))
print(response)

messages = [ChatMessage(role="user", content="...")]

response = llm.chat(messages)
print(response)

response = asyncio.run(llm.achat(messages))
print(response)
```

### Function Calling

```python
from llama_index.llms.siliconflow import SiliconFlow

llm = SiliconFlow(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
)
tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Compute the sum of two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "int",
                        "description": "A number",
                    },
                    "b": {
                        "type": "int",
                        "description": "A number",
                    },
                },
                "required": ["a", "b"],
            },
        },
    },
    ...,
]
response = llm.complete("...", tools=tools)
print(llm.get_tool_calls_from_response(response))

# output
# [ToolSelection(tool_id='...', tool_name='add', tool_kwargs={'a': x, 'b': x})]
```
