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

- Cosine vector distance support depends on the OceanBase version. If cosine is not supported, use inner product or Euclidean distance instead (Euclidean is the default).
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
    include_sparse=False,
    include_fulltext=False,
)
```

## Sparse / Fulltext / Hybrid Search

Enable sparse and fulltext support at initialization:

```python
oceanbase = OceanBaseVectorStore(
    client=client,
    dim=1536,
    drop_old=True,
    normalize=True,
    include_sparse=True,
    include_fulltext=True,
)
```

Use `VectorStoreQueryMode` for sparse, fulltext, or hybrid search. Sparse and fulltext
queries are passed via keyword arguments:

```python
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)

# sparse search
q = VectorStoreQuery(mode=VectorStoreQueryMode.SPARSE, similarity_top_k=5)
result = oceanbase.query(q, sparse_query={0: 1.0, 10: 0.5})

# fulltext search
q = VectorStoreQuery(
    mode=VectorStoreQueryMode.TEXT_SEARCH, query_str="oceanbase"
)
result = oceanbase.query(q)

# hybrid search
q = VectorStoreQuery(
    mode=VectorStoreQueryMode.HYBRID,
    query_embedding=[...],
    query_str="oceanbase",
    similarity_top_k=5,
)
result = oceanbase.query(q, sparse_query={0: 1.0})
```
