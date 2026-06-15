# LlamaIndex Llms Integration: TokenMix

[TokenMix](https://tokenmix.ai) is an OpenAI-compatible API gateway that exposes
DeepSeek, Qwen, Kimi, GLM, MiniMax and other models through a single endpoint
(`https://api.tokenmix.ai/v1`) and one API key.

### Installation

```bash
%pip install llama-index-llms-tokenmix
```

### Select Model

Browse the full model catalog at https://tokenmix.ai/models and use the full model
name (for example `deepseek/deepseek-v4-pro`).

### Basic usage

```py
from llama_index.llms.tokenmix import TokenMix

# Set your API key (or export TOKENMIX_API_KEY)
api_key = "Your API KEY"

# Call complete function
response = TokenMix(
    model="deepseek/deepseek-v4-pro", api_key=api_key
).complete("who are you")
print(response)

# Call chat with a list of messages
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(role="user", content="who are you"),
]

response = TokenMix(
    model="deepseek/deepseek-v4-pro", api_key=api_key
).chat(messages)
print(response)
```

### Streaming

```py
from llama_index.llms.tokenmix import TokenMix
from llama_index.core.llms import ChatMessage

llm = TokenMix(model="deepseek/deepseek-v4-pro", api_key="Your API KEY")

# Using stream_complete endpoint
response = llm.stream_complete("who are you")
for r in response:
    print(r.delta, end="")

# Using stream_chat endpoint
messages = [
    ChatMessage(role="user", content="who are you"),
]

response = llm.stream_chat(messages)
for r in response:
    print(r.delta, end="")
```

### Function Calling

```py
from datetime import datetime
from llama_index.core.tools import FunctionTool
from llama_index.llms.tokenmix import TokenMix


def get_current_time() -> dict:
    """Get the current time"""
    return {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


llm = TokenMix(model="deepseek/deepseek-v4-pro", api_key="Your API KEY")
tool = FunctionTool.from_defaults(fn=get_current_time)
response = llm.predict_and_call([tool], "What is the current time?")
print(response)
```

### TokenMix Documentation

API Documentation: https://tokenmix.ai/docs
