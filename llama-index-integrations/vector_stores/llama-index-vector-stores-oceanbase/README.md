# LlamaIndex Vector-Stores Integration: Oceanbase

[OceanBase Database](https://github.com/oceanbase/oceanbase) is a distributed relational database.
It is developed entirely by Ant Group. The OceanBase Database is built on a common server cluster.
Based on the Paxos protocol and its distributed structure, the OceanBase Database provides high availability and linear scalability.

OceanBase currently has the ability to store vectors. Users can easily perform the following operations with SQL:

- Create a table containing vector type fields;
- Create a vector index table based on the HNSW algorithm;
- Perform vector approximate nearest neighbor queries;
- ...

## Limitations

The vector storage capability of OceanBase is still being enhanced, and currently has the following limitations:

- Cosine vector distance function is not yet supported for vector index. Currently only inner product distance and Euclidean distance are supported, and Euclidean distance is used by default.
- It should be noted that `OceanBase` currently only supports post-filtering (i.e., filtering based on metadata after performing an approximate nearest neighbor search).

## Install dependencies

We use `pyobvector` to integrate OceanBase vector store into LlamaIndex.
So it is necessary to install it with `pip install pyobvector` before starting.

## Setup OceanBase

We recommend using Docker to deploy OceanBase:

```shell
docker run --name=ob433 -e MODE=slim -p 2881:2881 -d oceanbase/oceanbase-ce:4.3.3.0-100000142024101215
```

## Usage

```sh
%pip install llama-index-vector-stores-oceanbase
%pip install llama-index
# choose dashscope as embedding and llm model, your can also use default openai or other model to test
%pip install llama-index-embeddings-dashscope
%pip install llama-index-llms-dashscope
```

```python
from llama_index.vector_stores.oceanbase import OceanBaseVectorStore
from pyobvector import ObVecClient

client = ObVecClient()

client.perform_raw_text_sql(
    "ALTER SYSTEM ob_vector_memory_limit_percentage = 30"
)

# Initialize OceanBaseVectorStore
oceanbase = OceanBaseVectorStore(
    client=client,
    dim=1536,
    drop_old=True,
    normalize=True,
)
```
