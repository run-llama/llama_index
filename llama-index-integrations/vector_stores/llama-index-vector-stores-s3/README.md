# S3VectorStore Integration

This is a vector store integration for LlamaIndex that uses S3Vectors.

[Find out more about S3Vectors](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-getting-started.html).

This notebook will assume that you have already created a S3 vector bucket (and possibly also an index).

## Installation

```bash
pip install llama-index-vector-stores-s3
```

## Usage

### Creating the vector store object

You can create a new vector index in an existing S3 bucket.

```python
from llama_index.vector_stores.s3 import S3VectorStore
import boto3

vector_store = S3VectorStore.create_index_from_bucket(
    # S3 bucket name or ARN
    bucket_name_or_arn="my-vector-bucket",
    # Name for the new index
    index_name="my-index",
    # Vector dimension (e.g., 1536 for OpenAI embeddings)
    dimension=1536,
    # Distance metric: "cosine", "euclidean", etc.
    distance_metric="cosine",
    # Data type for vectors
    data_type="float32",
    # Batch size for inserting vectors (max 500)
    insert_batch_size=500,
    # Metadata keys that won't be filterable
    non_filterable_metadata_keys=["custom_field"],
    # Optional: provide a boto3 session for custom AWS configuration
    # sync_session=boto3.Session(region_name="us-west-2"),
)
```

Or, you can use an existing vector index in an existing S3 bucket.

```python
from llama_index.vector_stores.s3 import S3VectorStore
import boto3

vector_store = S3VectorStore(
    # Index name or ARN
    index_name_or_arn="my-index",
    # S3 bucket name or ARN
    bucket_name_or_arn="my-vector-bucket",
    # Data type for vectors (must match index)
    data_type="float32",
    # Distance metric (must match index)
    distance_metric="cosine",
    # Batch size for inserting vectors (max 500)
    insert_batch_size=500,
    # Optional: specify metadata field containing text if you already have a populated index
    text_field="content",
    # Optional: provide a boto3 session for custom AWS configuration
    # sync_session=boto3.Session(region_name="us-west-2"),
)
```

### Using the vector store with an index

Once you have a vector store, you can use it with an index:

```python
from llama_index.core import VectorStoreIndex, StorageContext

# Create a new index
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=StorageContext.from_defaults(vector_store=vector_store),
    # optional: set the embed model
    # embed_model=embed_model,
)

# Or reload from an existing index
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    # optional: set the embed model
    # embed_model=embed_model,
)
```

### Using the vector store directly

You can also use the vector store directly:

```python
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

# requires pip install llama-index-embeddings-openai
from llama_index.embeddings.openai import OpenAIEmbedding

# embed nodes
nodes = [
    TextNode(text="Hello, world!"),
    TextNode(text="Hello, world! 2"),
]

embed_model = OpenAIEmbedding(model="text-embedding-3-small")
embeddings = embed_model.get_text_embedding_batch(nodes)
for node, embedding in zip(nodes, embeddings):
    node.embedding = embedding

# add nodes to the vector store
vector_store.add(nodes)

# query the vector store
query = VectorStoreQuery(
    query_embedding=embed_model.get_query_embedding("Hello, world!"),
    similarity_top_k=2,
)
results = vector_store.query(query)
print(results.nodes)
```
