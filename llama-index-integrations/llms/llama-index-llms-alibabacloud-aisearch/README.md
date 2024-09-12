# LlamaIndex Llms Integration: Alibabacloud_Aisearch

## Installation

```
pip install llama-index-llms-alibabacloud-aisearch
```

## Usage

For further details, please visit [text-generation-api-details](https://help.aliyun.com/zh/open-search/search-platform/developer-reference/text-generation-api-details).

You can specify the `endpoint` and `aisearch_api_key` in the constructor, or set the environment variables `AISEARCH_ENDPOINT` and `AISEARCH_API_KEY`.

```python
from llama_index.llms.alibabacloud_aisearch import AlibabaCloudAISearchLLM
from llama_index.core import ChatPromptTemplate

llm = AlibabaCloudAISearchLLM(service_id="ops-qwen-turbo")
prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are helpful assistant."), ("user", "{prompt}")]
)
messages = prompt_template.format_messages(prompt="<your question>")
print(llm.chat(messages))
```
