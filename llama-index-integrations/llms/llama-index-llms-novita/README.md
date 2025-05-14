# LlamaIndex Llms Integration: NovitaAI

### Installation

```bash
%pip install llama-index-llms-novita
```

### Select Model

Large Language Models: https://novita.ai/llm-api?utm_source=github_llama_index&utm_medium=github_readme&utm_campaign=link

### Basic usage

```py
# Import NovitaAI
from llama_index.llms.novita import NovitaAI

# Set your API key
api_key = "Your API KEY"

# Call complete function
response = NovitaAI(
    model="meta-llama/llama-3.1-8b-instruct", api_key=api_key
).complete("who are you")
print(response)

# Call complete with stop
response = NovitaAI(
    model="meta-llama/llama-3.1-8b-instruct", api_key=api_key
).complete(prompt="who are you", stop=["AI"])
print(response)

# Call chat with a list of messages
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(role="user", content="who are you"),
]

response = NovitaAI(
    model="meta-llama/llama-3.1-8b-instruct", api_key=api_key
).chat(messages)
print(response)
```

### Streaming: Using stream endpoint

```py
from llama_index.llms.novita import NovitaAI

# Set your API key
api_key = "Your API KEY"

llm = NovitaAI(model="meta-llama/llama-3.1-8b-instruct", api_key=api_key)

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
from llama_index.llms.novita import NovitaAI

# Set your API key
api_key = "Your API KEY"


def get_current_time() -> dict:
    """Get the current time"""
    return {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


llm = NovitaAI(model="deepseek/deepseek_v3", api_key=api_key)
tool = FunctionTool.from_defaults(fn=get_current_time)
response = llm.predict_and_call([tool], "What is the current time?")
print(response)
```

### NovitaAI Documentation

API Documentation: https://novita.ai/docs/guides/llm-api
