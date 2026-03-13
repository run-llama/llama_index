# Endee LlamaIndex Integration

Build powerful RAG applications with Endee vector database and LlamaIndex.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Testing locally](#testing-locally)
3. [Setting up Credentials](#2-setting-up-endee-and-openai-credentials)
4. [Creating Sample Documents](#3-creating-sample-documents)
5. [Setting up Endee with LlamaIndex](#4-setting-up-endee-with-llamaindex)
6. [Creating a Vector Index](#5-creating-a-vector-index-from-documents)
7. [Basic Retrieval](#6-basic-retrieval-with-query-engine)
8. [Using Metadata Filters](#7-using-metadata-filters)
9. [Advanced Filtering](#8-advanced-filtering-with-multiple-conditions)
10. [Custom Retriever Setup](#9-custom-retriever-setup)
11. [Custom Retriever with Query Engine](#10-using-a-custom-retriever-with-a-query-engine)
12. [Direct VectorStore Querying](#11-direct-vectorstore-querying)
13. [Saving and Loading Indexes](#12-saving-and-loading-indexes)
14. [Cleanup](#13-cleanup)

---

## 1. Installation

Get started by installing the required package.

### Basic Installation (Dense-only search)

```bash
pip install endee-llamaindex
```

> **Note:** This will automatically install `endee` and `llama-index` as dependencies.

### Full Installation (with Hybrid Search support)

For hybrid search capabilities (dense + sparse vectors), install with the `hybrid` extra:

```bash
pip install endee-llamaindex[hybrid]
```

This includes FastEmbed for sparse vector encoding (SPLADE, BM25, etc.).

### GPU-Accelerated Hybrid Search

For GPU-accelerated sparse encoding:

```bash
pip install endee-llamaindex[hybrid-gpu]
```

### All Features

To install all optional dependencies:

```bash
pip install endee-llamaindex[all]
```

### Installation Options Summary

| Installation | Use Case | Includes |
|--------------|----------|----------|
| `pip install endee-llamaindex` | Dense vector search only | Core dependencies |
| `pip install endee-llamaindex[hybrid]` | Dense + sparse hybrid search | + FastEmbed (CPU) |
| `pip install endee-llamaindex[hybrid-gpu]` | GPU-accelerated hybrid search | + FastEmbed (GPU) |
| `pip install endee-llamaindex[all]` | All features | All optional deps |

---

## Testing locally

From the project root:

```bash
python -m venv env && source env/bin/activate   # optional
pip install -e .
pip install pytest sentence-transformers huggingface-hub
export ENDEE_API_TOKEN="your-endee-api-token"   # or set in llama-index/test_cases/setup_class.py

cd llama-index/test_cases && PYTHONPATH=.. python -m pytest . -v
```

See [TESTING.md](TESTING.md) for more options and single-test runs.

---

## 2. Setting up Endee and OpenAI credentials

Configure your API credentials for Endee and OpenAI.

```python
import os
from llama_index.embeddings.openai import OpenAIEmbedding

# Set API keys
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
endee_api_token = "your-endee-api-token"
```

> **Tip:** Store your API keys in environment variables for production use.

---

## 3. Creating Sample Documents

Create documents with metadata for filtering and organization.

```python
from llama_index.core import Document

# Create sample documents with different categories and metadata
documents = [
    Document(
        text="Python is a high-level, interpreted programming language known for its readability and simplicity.",
        metadata={"category": "programming", "language": "python", "difficulty": "beginner"}
    ),
    Document(
        text="JavaScript is a scripting language that enables interactive web pages and is an essential part of web applications.",
        metadata={"category": "programming", "language": "javascript", "difficulty": "intermediate"}
    ),
    Document(
        text="Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience.",
        metadata={"category": "ai", "field": "machine_learning", "difficulty": "advanced"}
    ),
    Document(
        text="Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
        metadata={"category": "ai", "field": "deep_learning", "difficulty": "advanced"}
    ),
    Document(
        text="Vector databases are specialized database systems designed to store and query high-dimensional vectors for similarity search.",
        metadata={"category": "database", "type": "vector", "difficulty": "intermediate"}
    ),
    Document(
        text="Endee is a vector database that provides secure and private vector search capabilities.",
        metadata={"category": "database", "type": "vector", "product": "endee", "difficulty": "intermediate"}
    )
]

print(f"Created {len(documents)} sample documents")
```

**Output:**
```
Created 6 sample documents
```

---

## 4. Setting up Endee with LlamaIndex

Initialize the Endee vector store and connect it to LlamaIndex.

```python
from llama-index import EndeeVectorStore
from llama_index.core import StorageContext
import time

# Create a unique index name with timestamp to avoid conflicts
timestamp = int(time.time())
index_name = f"llamaindex_demo_{timestamp}"

# Set up the embedding model
embed_model = OpenAIEmbedding()

# Get the embedding dimension
dimension = 1536  # OpenAI's default embedding dimension

# Initialize the Endee vector store
vector_store = EndeeVectorStore.from_params(
    api_token=endee_api_token,
    index_name=index_name,
    dimension=dimension,
    space_type="cosine",  # Can be "cosine", "l2", or "ip"
    precision="float16"  # Options: "binary", "float16", "float32", "int16", "int8" (default: "float16")
)

# Create storage context with our vector store
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print(f"Initialized Endee vector store with index: {index_name}")
```

### Configuration Options

| Parameter | Description | Options |
|-----------|-------------|---------|
| `space_type` | Distance metric for similarity | `cosine`, `l2`, `ip` |
| `dimension` | Vector dimension | Must match embedding model |
| `precision` | Index precision setting | `"binary"`, `"float16"` (default), `"float32"`, `"int16"`, `"int8"` |
| `batch_size` | Vectors per API call | Default: `100` |
| `hybrid` | Enable hybrid search (dense + sparse) | Default: `False` |
| `M` | Optional HNSW M parameter (bi-directional links) | Optional (backend default if not specified) |
| `ef_con` | Optional HNSW ef_construction parameter | Optional (backend default if not specified) |

### Hybrid Search and Sparse Models

When you enable hybrid search by providing a positive `sparse_dim` and a `model_name`, the vector store automatically computes sparse (bag-of-words‑style) vectors in addition to dense vectors.

- **Sparse dimension (`sparse_dim`)**:
  - For the built-in SPLADE models, the recommended `sparse_dim` is **30522** (matching the model vocabulary size).
  - For dense‑only search, omit `sparse_dim` (or set it to `0`).
- **Supported sparse models (`model_name`)**:
  - `"splade_pp"` → `prithivida/Splade_PP_en_v1` (SPLADE++)
  - `"splade_cocondenser"` → `naver/splade-cocondenser-ensembledistil`

Example hybrid configuration:

```python
vector_store = EndeeVectorStore.from_params(
    api_token=endee_api_token,
    index_name=index_name,
    dimension=dimension,        # dense dimension (e.g., 1536 for OpenAI)
    space_type="cosine",
    precision="float16",
    hybrid=True,
    sparse_dim=30522,           # sparse dimension for SPLADE models
    model_name="splade_pp",     # or "splade_cocondenser"
)
```

---

## 5. Creating a Vector Index from Documents

Build a searchable vector index from your documents.

```python
from llama_index.core import VectorStoreIndex

# Create a vector index
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model
)

print("Vector index created successfully")
```

**Output:**
```
Vector index created successfully
```

---

## 6. Basic Retrieval with Query Engine

Create a query engine and perform semantic search.

```python
# Create a query engine
query_engine = index.as_query_engine()

# Ask a question
response = query_engine.query("What is Python?")

print("Query: What is Python?")
print("Response:")
print(response)
```

**Example Output:**
```
Query: What is Python?
Response:
Python is a high-level, interpreted programming language known for its readability and simplicity.
```

---

## 7. Using Metadata Filters

Filter search results based on document metadata.

```python
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator

# Create a filtered retriever to only search within AI-related documents
ai_filter = MetadataFilter(key="category", value="ai", operator=FilterOperator.EQ)
ai_filters = MetadataFilters(filters=[ai_filter])

# Create a filtered query engine
filtered_query_engine = index.as_query_engine(filters=ai_filters)

# Ask a general question but only using AI documents
response = filtered_query_engine.query("What is learning from data?")

print("Filtered Query (AI category only): What is learning from data?")
print("Response:")
print(response)
```

### Available Filter Operators

| Operator | Description | Backend Symbol | Example |
|----------|-------------|----------------|---------|
| `FilterOperator.EQ` | Equal to | `$eq` | `rating == 5` |
| `FilterOperator.IN` | In list | `$in` | `category in ["ai", "ml"]` |


> **Important Notes:**
> - Currently, the Endee LlamaIndex integration only supports **EQ** and **IN** metadata filters.
> - Range-style operators (LT, LTE, GT, GTE) are **not** supported in this adapter.

### Filter Examples

Here are practical examples showing how to use the supported filter operators:

```python
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator

# Example 1: Equal to (EQ)
# Find documents with rating equal to 5
rating_filter = MetadataFilter(key="rating", value=5, operator=FilterOperator.EQ)
filters = MetadataFilters(filters=[rating_filter])
# Backend: {"rating": {"$eq": 5}}

# Example 2: In list (IN)
# Find documents in AI or ML categories
category_filter = MetadataFilter(key="category", value=["ai", "ml"], operator=FilterOperator.IN)
filters = MetadataFilters(filters=[category_filter])
# Backend: {"category": {"$in": ["ai", "ml"]}}

# Example 3: Combined filters (AND logic)
# Find AI documents with rating equal to 5
filters = MetadataFilters(filters=[
    MetadataFilter(key="category", value="ai", operator=FilterOperator.EQ),
    MetadataFilter(key="rating", value=5, operator=FilterOperator.EQ)
])
# Backend: [{"category": {"$eq": "ai"}}, {"rating": {"$eq": 5}}]

# Create a query engine with filters
filtered_engine = index.as_query_engine(filters=filters)
response = filtered_engine.query("What is machine learning?")
```

---

## 8. Advanced Filtering with Multiple Conditions

Combine multiple metadata filters for precise results.

```python
# Create a more complex filter: database category AND intermediate difficulty
category_filter = MetadataFilter(key="category", value="database", operator=FilterOperator.EQ)
difficulty_filter = MetadataFilter(key="difficulty", value="intermediate", operator=FilterOperator.EQ)

complex_filters = MetadataFilters(filters=[category_filter, difficulty_filter])

# Create a query engine with the complex filters
complex_filtered_engine = index.as_query_engine(filters=complex_filters)

# Query with the complex filters
response = complex_filtered_engine.query("Tell me about databases")

print("Complex Filtered Query (database category AND intermediate difficulty): Tell me about databases")
print("Response:")
print(response)
```

> **Note:** Multiple filters are combined with AND logic by default.

---

## 9. Custom Retriever Setup

Create a custom retriever for fine-grained control over the retrieval process.

```python
from llama_index.core.retrievers import VectorIndexRetriever

# Create a retriever with custom parameters
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3,  # Return top 3 most similar results
    filters=ai_filters   # Use our AI category filter from before
)

# Retrieve nodes for a query
nodes = retriever.retrieve("What is deep learning?")

print(f"Retrieved {len(nodes)} nodes for query: 'What is deep learning?' (with AI category filter)")
print("\nRetrieved content:")
for i, node in enumerate(nodes):
    print(f"\nNode {i+1}:")
    print(f"Text: {node.node.text}")
    print(f"Metadata: {node.node.metadata}")
    print(f"Score: {node.score:.4f}")
```

**Example Output:**
```
Retrieved 2 nodes for query: 'What is deep learning?' (with AI category filter)

Node 1:
Text: Deep learning is part of a broader family of machine learning methods...
Metadata: {'category': 'ai', 'field': 'deep_learning', 'difficulty': 'advanced'}
Score: 0.8934

Node 2:
Text: Machine learning is a subset of artificial intelligence...
Metadata: {'category': 'ai', 'field': 'machine_learning', 'difficulty': 'advanced'}
Score: 0.7821
```

---

## 10. Using a Custom Retriever with a Query Engine

Combine your custom retriever with a query engine for enhanced control.

```python
from llama_index.core.query_engine import RetrieverQueryEngine

# Create a query engine with our custom retriever
custom_query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    verbose=True  # Enable verbose mode to see the retrieved nodes
)

# Query using the custom retriever query engine
response = custom_query_engine.query("Explain the difference between machine learning and deep learning")

print("\nFinal Response:")
print(response)
```

---

## 11. Direct VectorStore Querying

Query the Endee vector store directly, bypassing the LlamaIndex query engine.

```python
from llama_index.core.vector_stores.types import VectorStoreQuery

# Generate an embedding for our query
query_text = "What are vector databases?"
query_embedding = embed_model.get_text_embedding(query_text)

# Create a VectorStoreQuery
vector_store_query = VectorStoreQuery(
    query_embedding=query_embedding,
    similarity_top_k=2,
    filters=MetadataFilters(filters=[MetadataFilter(key="category", value="database", operator=FilterOperator.EQ)])
)

# Execute the query directly on the vector store
query_result = vector_store.query(vector_store_query)

print(f"Direct VectorStore query: '{query_text}'")
print(f"Retrieved {len(query_result.nodes)} results with database category filter:")
for i, (node, score) in enumerate(zip(query_result.nodes, query_result.similarities)):
    print(f"\nResult {i+1}:")
    print(f"Text: {node.text}")
    print(f"Metadata: {node.metadata}")
    print(f"Similarity score: {score:.4f}")
```

> **Tip:** Direct querying is useful when you need raw results without LLM processing.

---

## 12. Saving and Loading Indexes

Reconnect to your index in future sessions. Your vectors are stored in the cloud.

```python
# To reconnect to an existing index in a future session:
def reconnect_to_index(api_token, index_name):
    # Initialize the vector store with existing index
    vector_store = EndeeVectorStore.from_params(
        api_token=api_token,
        index_name=index_name
    )
    
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Load the index
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=OpenAIEmbedding()
    )
    
    return index

# Example usage
reconnected_index = reconnect_to_index(endee_api_token, index_name)
query_engine = reconnected_index.as_query_engine()
response = query_engine.query("What is Endee?")
print(response)

print(f"To reconnect to this index in the future, use:\n")
print(f"API Token: {endee_api_token}")
print(f"Index Name: {index_name}")
```

> **Important:** Save your `index_name` to reconnect to your data later.

---

## 13. Cleanup

Delete the index when you're done to free up resources.

```python
# Uncomment to delete your index
# endee.delete_index(index_name)
# print(f"Index {index_name} deleted")
```

> **Warning:** Deleting an index permanently removes all stored vectors and cannot be undone.

---

## Quick Reference

### EndeeVectorStore Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `api_token` | `str` | Your Endee API token | Required |
| `index_name` | `str` | Name of the index | Required |
| `dimension` | `int` | Vector dimension | Required |
| `space_type` | `str` | Distance metric (`"cosine"`, `"l2"`, `"ip"`) | `"cosine"` |
| `precision` | `str` | Index precision (`"binary"`, `"float16"`, `"float32"`, `"int16"`, `"int8"`) | `"float16"` |
| `batch_size` | `int` | Vectors per API call | `100` |
| `hybrid` | `bool` | Enable hybrid search (dense + sparse vectors) | `False` |
| `sparse_dim` | `int` | Sparse dimension for hybrid index | `None` |
| `model_name` | `str` | Model name for sparse embeddings (e.g., `'splade_pp'`, `'bert_base'`) | `None` |
| `M` | `int` | Optional HNSW M parameter (bi-directional links per node) | `None` (backend default) |
| `ef_con` | `int` | Optional HNSW ef_construction parameter | `None` (backend default) |
| `prefilter_cardinality_threshold` | `int` | Switches from HNSW filtered search to brute-force prefiltering when filter matches ≤ this many vectors (range: 1,000–1,000,000) | `None` (server default: 10,000) |
| `filter_boost_percentage` | `int` | Expands the HNSW candidate pool by this percentage when a filter is active to compensate for filtered-out results (range: 0–100) | `None` (server default: 0) |

### Distance Metrics

| Metric | Best For |
|--------|----------|
| `cosine` | Text embeddings, normalized vectors |
| `l2` | Image features, spatial data |
| `ip` | Recommendation systems, dot product similarity |

### Precision Settings

The `precision` parameter controls the vector storage format and affects memory usage and search performance:

| Precision | Description | Use Case |
|-----------|-------------|----------|
| `"float32"` | Full precision floating point | Maximum accuracy, higher memory usage |
| `"float16"` | Half precision floating point | Balanced accuracy and memory (default) |
| `"binary"` | Binary vectors | Extremely compact, best for binary embeddings |
| `"int8"` | 8-bit integer quantization | High compression, good accuracy |
| `"int16"` | 16-bit integer quantization | Better accuracy than int8, moderate compression |

### HNSW Parameters (Optional)

HNSW (Hierarchical Navigable Small World) parameters control index construction and search quality. These are **optional** - if not provided, the Endee backend uses optimized defaults.

| Parameter | Description | Impact |
|-----------|-------------|--------|
| `M` | Number of bi-directional links per node | Higher M = better recall, more memory |
| `ef_con` | Size of dynamic candidate list during construction | Higher ef_con = better quality, slower indexing |

**Example with custom HNSW parameters:**

```python
vector_store = EndeeVectorStore.from_params(
    api_token="your-token",
    index_name="custom_index",
    dimension=384,
    space_type="cosine",
    M=32,           # Optional: custom M value
    ef_con=256      # Optional: custom ef_construction
)
```

**Note:** Only specify M and ef_con if you need to fine-tune performance. The backend defaults work well for most use cases.

### Filter Tuning Parameters (Optional)

When using filtered queries, two optional parameters let you tune the trade-off between search speed and recall. Pass them directly to `vector_store.query()`.

#### `prefilter_cardinality_threshold`

Controls when the search strategy switches from **HNSW filtered search** (fast, graph-based) to **brute-force prefiltering** (exhaustive scan on the matched subset).

| Value | Behavior |
|-------|----------|
| `1_000` | Prefilter only for very selective filters — minimum value |
| `10_000` | Prefilter only when the filter matches ≤10,000 vectors **(server default)** |
| `1_000_000` | Prefilter for almost all filtered searches — maximum value |

When very few vectors match your filter, HNSW may struggle to find enough valid candidates through graph traversal. Prefiltering the matched subset directly is faster and more accurate in that case. Raising the threshold means prefiltering kicks in more often; lowering it favors HNSW graph search.

#### `filter_boost_percentage`

When using HNSW filtered search, some candidates explored during graph traversal are discarded by the filter, which can leave you with fewer results than `top_k`. `filter_boost_percentage` compensates by expanding the internal candidate pool before filtering is applied.

- `0` → no boost, standard candidate pool size **(server default)**
- `20` → fetch 20% more candidates internally before applying the filter
- Maximum: `100` (doubles the candidate pool)

**Example:**

```python
from llama_index.core.vector_stores.types import VectorStoreQuery

result = vector_store.query(
    VectorStoreQuery(
        query_embedding=[...],
        similarity_top_k=10,
    ),
    prefilter_cardinality_threshold=5_000,  # switch to brute-force for small match sets
    filter_boost_percentage=30,             # boost candidates for HNSW filtered search
)
```

> **Tip:** Start with the defaults. If filtered queries return fewer results than expected, try increasing `filter_boost_percentage`. If filtered queries are slow on selective filters, try lowering `prefilter_cardinality_threshold`.

---


