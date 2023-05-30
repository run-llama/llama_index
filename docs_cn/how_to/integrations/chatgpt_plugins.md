# ChatGPT插件集成

**注意**：这是一个正在进行中的工作，敬请期待更多令人兴奋的更新！

## ChatGPT检索插件集成

[OpenAI ChatGPT检索插件](https://github.com/openai/chatgpt-retrieval-plugin)提供了一个集中的API规范，用于任何文档存储系统与ChatGPT进行交互。由于它可以部署在任何服务上，这意味着越来越多的文档检索服务将实施此规范；这使它们不仅可以与ChatGPT交互，还可以与任何可能使用检索服务的LLM工具包交互。

LlamaIndex提供了与ChatGPT检索插件的多种集成。

###从LlamaHub加载数据到ChatGPT检索插件

ChatGPT检索插件为用户定义了一个“/ upsert”端点，用于加载文档。这提供了与LlamaHub的自然集成点，它提供来自各种API和文档格式的65个以上的数据加载器。

以下是一个示例代码片段，显示如何将LlamaHub中的文档加载到“/ upsert”所期望的JSON格式：

```python
from llama_index import download_loader, Document
from typing import Dict, List
import json

# download loader, load documents
SimpleWebPageReader = download_loader("SimpleWebPageReader")
loader = SimpleWebPageReader(html_to_text=True)
url = "http://www.paulgraham.com/worked.html"
documents = loader.load_data(urls=[url])

# Convert LlamaIndex Documents to JSON format
def dump_docs_to_json(documents: List[Document], out_path: str) -> Dict:
    """Convert LlamaIndex Documents to JSON format and save it."""
    result_json = []
    for doc in documents:
        cur_dict = {
            "text": doc.get_text(),
            "id": doc.get_doc_id(),
            # NOTE: feel free to customize the other fields as you wish
            # fields taken from https://github.com/openai/chatgpt-retrieval-plugin/tree/main/scripts/process_json#usage
            # "source": ...,
            # "source_id": ...,
            # "url": url,
            # "created_at": ...,
            # "author": "Paul Graham",
        }
        result_json.append(cur_dict)

    json.dump(result_json, open(out_path, 'w'))

```

有关更多详细信息，请查看[完整的示例笔记本](https://github.com/jerryjliu/llama_index/blob/main/examples/chatgpt_plugin/ChatGPT_Retrieval_Plugin_Upload.ipynb)。

### ChatGPT检索插件数据加载器

可以在LlamaHub上访问[ChatGPT检索插件数据加载器](https://llamahub.ai/l/chatgpt_plugin)。

它允许您轻松从任何docstore加载数据，该docstore实现了ChatGPT检索插件规范。ChatGPT Retrieval Plugin Index

ChatGPT Retrieval Plugin Index可以让您轻松地在任何文档上构建一个向量索引，存储由实现ChatGPT端点的文档存储支持。

注意：此索引是一个向量索引，允许top-k检索。

示例代码：

```python
from llama_index.indices.vector_store import ChatGPTRetrievalPluginIndex
from llama_index import SimpleDirectoryReader
import os

# 加载文档
documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()

# 构建索引
bearer_token = os.getenv("BEARER_TOKEN")
# 不使用元数据过滤器初始化
index = ChatGPTRetrievalPluginIndex(
    documents,
    endpoint_url="http://localhost:8000",
    bearer_token=bearer_token,
)

# 查询索引
query_engine = vector_index.as_query_engine(
    similarity_top_k=3,
    response_mode="compact",
)
response = query_engine.query("What did the author do growing up?")

```

更多详情，请查看[完整的示例笔记本](https://github.com/jerryjliu/llama_index/blob/main/examples/chatgpt_plugin/ChatGPTRetrievalPluginIndexDemo.ipynb)。