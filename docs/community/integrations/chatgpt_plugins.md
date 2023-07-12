# ChatGPT Plugin Integrations

**NOTE**: This is a work-in-progress, stay tuned for more exciting updates on this front! 

## ChatGPT Retrieval Plugin Integrations

The [OpenAI ChatGPT Retrieval Plugin](https://github.com/openai/chatgpt-retrieval-plugin)
offers a centralized API specification for any document storage system to interact 
with ChatGPT. Since this can be deployed on any service, this means that more and more
document retrieval services will implement this spec; this allows them to not only
interact with ChatGPT, but also interact with any LLM toolkit that may use 
a retrieval service.

LlamaIndex provides a variety of integrations with the ChatGPT Retrieval Plugin.

### Loading Data from LlamaHub into the ChatGPT Retrieval Plugin

The ChatGPT Retrieval Plugin defines an `/upsert` endpoint for users to load
documents. This offers a natural integration point with LlamaHub, which offers
over 65 data loaders from various API's and document formats.

Here is a sample code snippet of showing how to load a document from LlamaHub
into the JSON format that `/upsert` expects:

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

For more details, check out the [full example notebook](https://github.com/jerryjliu/llama_index/blob/main/examples/chatgpt_plugin/ChatGPT_Retrieval_Plugin_Upload.ipynb).

### ChatGPT Retrieval Plugin Data Loader

The ChatGPT Retrieval Plugin data loader [can be accessed on LlamaHub](https://llamahub.ai/l/chatgpt_plugin).

It allows you to easily load data from any docstore that implements the plugin API, into a LlamaIndex data structure.

Example code:

```python
from llama_index.readers import ChatGPTRetrievalPluginReader
import os

# load documents
bearer_token = os.getenv("BEARER_TOKEN")
reader = ChatGPTRetrievalPluginReader(
    endpoint_url="http://localhost:8000",
    bearer_token=bearer_token
)
documents = reader.load_data("What did the author do growing up?")

# build and query index
from llama_index import ListIndex
index = ListIndex(documents)
# set Logging to DEBUG for more detailed outputs
query_engine = vector_index.as_query_engine(
    response_mode="compact"
)
response = query_engine.query(
    "Summarize the retrieved content and describe what the author did growing up",
) 

```
For more details, check out the [full example notebook](https://github.com/jerryjliu/llama_index/blob/main/examples/chatgpt_plugin/ChatGPTRetrievalPluginReaderDemo.ipynb).

### ChatGPT Retrieval Plugin Index

The ChatGPT Retrieval Plugin Index allows you to easily build a vector index over any documents, with storage backed by a document store implementing the 
ChatGPT endpoint.

Note: this index is a vector index, allowing top-k retrieval.

Example code:

```python
from llama_index.indices.vector_store import ChatGPTRetrievalPluginIndex
from llama_index import SimpleDirectoryReader
import os

# load documents
documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()

# build index
bearer_token = os.getenv("BEARER_TOKEN")
# initialize without metadata filter
index = ChatGPTRetrievalPluginIndex(
    documents, 
    endpoint_url="http://localhost:8000",
    bearer_token=bearer_token,
)

# query index
query_engine = vector_index.as_query_engine(
    similarity_top_k=3,
    response_mode="compact",
)
response = query_engine.query("What did the author do growing up?")

```

For more details, check out the [full example notebook](https://github.com/jerryjliu/llama_index/blob/main/examples/chatgpt_plugin/ChatGPTRetrievalPluginIndexDemo.ipynb).