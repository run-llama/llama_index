# LlamaIndex Llms Integration: iFlytek Spark

iFlytek Spark exposes an OpenAI-compatible chat-completions API, so this integration builds on `OpenAILike`.

## Installation

```bash
pip install llama-index-llms-iflytek
```

## Usage

Get an API password from the [iFlytek open platform console](https://console.xfyun.cn/) and pass it as `api_key` (or set the `IFLYTEK_API_KEY` environment variable). Pick a model such as `generalv3.5`, `4.0Ultra` or `lite`.

```python
from llama_index.llms.iflytek import IFlytek

llm = IFlytek(model="4.0Ultra", api_key="your-api-password")

response = llm.complete("用一句话介绍你自己")
print(response)
```

Chat and streaming work the same way as any other `OpenAILike` model:

```python
from llama_index.core.llms import ChatMessage

messages = [ChatMessage(role="user", content="Hello")]
for chunk in llm.stream_chat(messages):
    print(chunk.delta, end="")
```

The default API base is `https://spark-api-open.xf-yun.com/v1`; override it with the `IFLYTEK_API_BASE` environment variable or the `api_base` argument if needed.
