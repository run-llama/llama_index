# LlamaIndex Vector_Stores Integration: Lindorm
- LindormVectorStore support pure vector search, search with metadata filtering, hybrid search, aysnc, etc. 
- Please refer to the [notebook](../../../docs/docs/examples/vector_stores/LindormDemo.ipynb) for usage of Lindorm as vector store in LlamaIndex.

# Example Usage

```sh
pip install llama-index
pip install opensearch-py
pip install llama-index-vector-stores-lindorm
```

```python
from llama_index.vector_stores.lindorm import (
    LindormVectorStore,
    LindormVectorClient,
)

# how to obtain an lindorm search instance:
# https://alibabacloud.com/help/en/lindorm/latest/create-an-instance

# how to access your lindorm search instance:
# https://www.alibabacloud.com/help/en/lindorm/latest/view-endpoints

# run curl commands to connect to and use LindormSearch:
# https://www.alibabacloud.com/help/en/lindorm/latest/connect-and-use-the-search-engine-with-the-curl-command

# lindorm instance info
host = "ld-bp******jm*******-proxy-search-pub.lindorm.aliyuncs.com"
port = 30070
username = 'your_username'
password = 'your_password'

# index to demonstrate the VectorStore impl
index_name = "lindorm_test_index"

# extenion param of lindorm search, number of cluster units to query; between 1 and method.parameters.nlist.
nprobe = "a number(string type)" 

# extenion param of lindorm search, usually used to improve recall accuracy, but it increases performance overhead; 
#   between 1 and 200; default: 10.
reorder_factor = "a number(string type)"

# LindormVectorClient encapsulates logic for a single index with vector search enabled
client = LindormVectorClient(
    host=host, 
    port=port,
    username=username, 
    password=password,
    index=index_name, 
    dimension=1536, # match with your embedding model
    nprobe=nprobe,
    reorder_factor=reorder_factor,
    # filter_type="pre_filter/post_filter(default)"
)

# initialize vector store
vector_store = LindormVectorStore(client)
```