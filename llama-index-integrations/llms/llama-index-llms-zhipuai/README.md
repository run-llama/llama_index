# LlamaIndex Llms Integration: ZhipuAI

### Installation

```bash
%pip install llama-index-llms-zhipuai
!pip install llama-index
```

### Basic usage

```py
# Import ZhipuAI
from llama_index.llms.zhipuai import ZhipuAI

# Set your API key
api_key = "Your API KEY"

# Call complete function
response = ZhipuAI(model="glm-4", api_key=api_key).complete("who are you")
print(response)

# Output
# I am an AI assistant named ZhiPuQingYan（智谱清言）, you can call me Xiaozhi🤖, which is developed based on the language model jointly trained by Tsinghua University KEG Lab and Zhipu AI Company in 2023. My job is to provide appropriate answers and support to users' questions and requests.

# Call complete with stop
response = ZhipuAI(model="glm-4", api_key=api_key).complete(
    prompt="who are you", stop=["Zhipu"]
)
print(response)

# Output
# I am an AI assistant named ZhiPuQingYan（智谱清言）, you can call me Xiaozhi🤖, which is developed based on the language model jointly trained by Tsinghua University KEG Lab and Zhipu

# Call chat with a list of messages
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(role="user", content="who are you"),
]

response = ZhipuAI(model="glm-4", api_key=api_key).chat(messages)
print(response)

# Output
# assistant: I am an AI assistant named ZhiPuQingYan（智谱清言）, you can call me Xiaozhi🤖, which is developed based on the language model jointly trained by Tsinghua University KEG Lab and Zhipu AI Company in 2023. My job is to provide appropriate answers and support to users' questions and requests.
```

### Streaming: Using stream endpoint

```py
from llama_index.llms.zhipuai import ZhipuAI

llm = ZhipuAI(model="glm-4", api_key=api_key)

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
from llama_index.llms.zhipuai import ZhipuAI

llm = ZhipuAI(model="glm-4", api_key="YOUR API KEY")
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_weather",
            "description": "Query the weather of the city provided by user",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City to query",
                    },
                },
                "required": ["city"],
            },
        },
    }
]
response = llm.complete(
    "help me to find the weather in Shanghai",
    tools=tools,
    tool_choice="auto",
)
print(llm.get_tool_calls_from_response(response))

# Output
# [ToolSelection(tool_id='call_9097928240216277928', tool_name='query_weather', tool_kwargs={'city': 'Shanghai'})]
```

### ZhipuAI Documentation

usage: https://bigmodel.cn/dev/howuse/introduction

api: https://bigmodel.cn/dev/api/normal-model/glm-4
