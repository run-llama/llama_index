# Migrating to Vertex AI Vector Search v2.0

This guide helps you migrate from Vertex AI Vector Search v1.0 (index/endpoint) to v2.0 (collections).

## Prerequisites

### 1. Install v2 Dependencies

```bash
pip install 'llama-index-vector-stores-vertexaivectorsearch[v2]'
```

### 2. Create a v2 Collection in Google Cloud

Before using v2, create a collection in the Google Cloud Console or via the API:

```python
from google.cloud import vectorsearch_v1beta

client = vectorsearch_v1beta.VectorSearchServiceClient()

collection = client.create_collection(
    parent=f"projects/{PROJECT_ID}/locations/{REGION}",
    collection_id="my-collection",
    collection={
        "display_name": "My Collection",
        "vector_config": {
            "vector_dimension": 768,  # Match your embedding dimension
            "distance_measure_type": "DOT_PRODUCT_DISTANCE",
        },
    },
)
```

---

## Migration Steps

### Step 1: Update Your Initialization Code

#### Before (v1)

```python
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore

vector_store = VertexAIVectorStore(
    project_id="my-project",
    region="us-central1",
    index_id="projects/my-project/locations/us-central1/indexes/123456",
    endpoint_id="projects/my-project/locations/us-central1/indexEndpoints/789012",
    gcs_bucket_name="my-staging-bucket",  # Required for batch updates
)
```

#### After (v2)

```python
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore

vector_store = VertexAIVectorStore(
    api_version="v2",  # Add this line
    project_id="my-project",
    region="us-central1",
    collection_id="my-collection"  # Replace index_id/endpoint_id with collection_id
    # No gcs_bucket_name needed!
)
```

### Step 2: Update Your Operations (No Changes Needed!)

The LlamaIndex interface remains the same. Your existing code works as-is:

```python
# Adding documents - SAME API
ids = vector_store.add(nodes)

# Querying - SAME API
results = vector_store.query(query)

# Deleting by document ID - SAME API
vector_store.delete(ref_doc_id="doc123")
```

### Step 3: Use New v2 Features (Optional)

v2 provides additional capabilities:

```python
# Delete specific nodes by ID (enhanced in v2)
vector_store.delete_nodes(node_ids=["node1", "node2", "node3"])

# Clear all data from collection (v2 only!)
vector_store.clear()
```

---

## Parameter Reference

### v1-Only Parameters (Remove for v2)

| Parameter         | Description                            |
| ----------------- | -------------------------------------- |
| `index_id`        | Vertex AI index resource name          |
| `endpoint_id`     | Vertex AI index endpoint resource name |
| `gcs_bucket_name` | GCS bucket for batch updates           |

### v2-Only Parameters (Add for v2)

| Parameter       | Description             |
| --------------- | ----------------------- |
| `api_version`   | Must be `"v2"`          |
| `collection_id` | Your v2 collection name |

### Shared Parameters (Work in Both)

| Parameter          | Description                              |
| ------------------ | ---------------------------------------- |
| `project_id`       | Google Cloud project ID                  |
| `region`           | Google Cloud region                      |
| `credentials_path` | Path to service account JSON (optional)  |
| `batch_size`       | Batch size for operations (default: 100) |

---

## Complete Example

### v2 with LlamaIndex VectorStoreIndex

```python
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore
from llama_index.embeddings.vertex import VertexTextEmbedding

# Set up embeddings
embed_model = VertexTextEmbedding(project="my-project", location="us-central1")

# Create v2 vector store
vector_store = VertexAIVectorStore(
    api_version="v2",
    project_id="my-project",
    region="us-central1",
    collection_id="my-collection",
)

# Create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create documents
documents = [
    Document(text="LlamaIndex is a data framework for LLM applications."),
    Document(
        text="Vertex AI Vector Search provides vector similarity search."
    ),
]

# Build index (automatically adds to vector store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is LlamaIndex?")
print(response)
```

### v2 Direct Vector Store Usage

```python
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore
from llama_index.embeddings.vertex import VertexTextEmbedding

# Set up
embed_model = VertexTextEmbedding(project="my-project", location="us-central1")

vector_store = VertexAIVectorStore(
    api_version="v2",
    project_id="my-project",
    region="us-central1",
    collection_id="my-collection",
)

# Create nodes with embeddings
text = "This is a sample document about machine learning."
embedding = embed_model.get_text_embedding(text)

node = TextNode(
    text=text,
    embedding=embedding,
    metadata={"category": "ML", "source": "example"},
)

# Add to vector store
ids = vector_store.add([node])
print(f"Added nodes: {ids}")

# Query
query_embedding = embed_model.get_query_embedding("machine learning")
query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=5)

results = vector_store.query(query)
for node, score in zip(results.nodes, results.similarities):
    print(f"Score: {score:.4f} - {node.text[:50]}...")
```

---

## Troubleshooting

### Error: "collection_id is required for v2"

You're using `api_version="v2"` but didn't provide `collection_id`:

```python
# Wrong
VertexAIVectorStore(api_version="v2", project_id="...", region="...")

# Correct
VertexAIVectorStore(
    api_version="v2",
    project_id="...",
    region="...",
    collection_id="my-collection",
)
```

### Error: "index_id is only valid for api_version='v1'"

You're mixing v1 and v2 parameters:

```python
# Wrong - can't use index_id with v2
VertexAIVectorStore(api_version="v2", collection_id="...", index_id="...")

# Correct - use only v2 parameters
VertexAIVectorStore(api_version="v2", collection_id="...")
```

### Error: "v2 operations require google-cloud-vectorsearch"

Install the v2 dependencies:

```bash
pip install 'llama-index-vector-stores-vertexaivectorsearch[v2]'
```

### Want to Force v1 Behavior?

Set the environment variable:

```bash
export VERTEX_AI_ENABLE_V2=false
```

---

## Key Differences Summary

| Feature                | v1                                        | v2                             |
| ---------------------- | ----------------------------------------- | ------------------------------ |
| **Architecture**       | Index + Endpoint                          | Collection                     |
| **Setup Complexity**   | Higher (create index, deploy to endpoint) | Lower (just create collection) |
| **GCS Bucket**         | Required for batch updates                | Not needed                     |
| **clear() method**     | Not supported                             | Supported                      |
| **Automatic Indexing** | Manual deployment                         | Automatic                      |
| **SDK**                | google-cloud-aiplatform                   | google-cloud-vectorsearch      |

---

## Need Help?

- [Vertex AI Vector Search v2 Documentation](https://cloud.google.com/vertex-ai/docs/vector-search/overview)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Report Issues](https://github.com/run-llama/llama_index/issues)
