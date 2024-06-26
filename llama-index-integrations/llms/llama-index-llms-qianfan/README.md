# LlamaIndex Llms Integration: Baidu Qianfan

Baidu Intelligent Cloud's Qianfan LLM Platform Chat Client: Implementing the CustomLLM Abstraction Class.

## Prerequisites:

1. Enable LLM Service: Before using the chat client, you need to activate the LLM service on the Qianfan LLM Platform console's online service page.

2. Create Access Key and Secret Key: Generate an Access Key and a Secret Key in the Security Authentication Center of the console.

## Initializing the Client

```
from llama_index.llms.qianfan import Qianfan

access_key = 'XXX'
secret_key = 'XXX'
model_name = 'ERNIE-Speed-8K'
endpoint_url = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie_speed'
context_window = 8192
llm = Qianfan(access_key, secret_key, model_name, endpoint_url, context_window)
```
