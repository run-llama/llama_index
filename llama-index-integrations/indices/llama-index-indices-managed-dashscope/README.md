# LlamaIndex Indices Integration: Managed-Dashscope

## Installation

```shell
pip install llama-index-indices-managed-dashscope
```

## Usage

```python
import os
from llama_index.core.schema import QueryBundle
from llama_index.readers.dashscope.base import DashScopeParse
from llama_index.readers.dashscope.utils import ResultType

os.environ["DASHSCOPE_API_KEY"] = "your_api_key_here"
os.environ["DASHSCOPE_WORKSPACE_ID"] = "your_workspace_here"

# init retriever from scratch
from llama_index.indices.managed.dashscope.retriever import (
    DashScopeCloudRetriever,
)


file_list = [
    # your files (accept doc, docx, pdf)
]

parse = DashScopeParse(result_type=ResultType.DASHCOPE_DOCMIND)
documents = parse.load_data(file_path=file_list)

# create a new index
index = DashScopeCloudIndex.from_documents(
    documents,
    "my_first_index",
    verbose=True,
)

# # connect to an existing index
# index = DashScopeCloudIndex("my_first_index")

retriever = index.as_retriever()
nodes = retriever.retrieve("test query")
print(nodes)
```
