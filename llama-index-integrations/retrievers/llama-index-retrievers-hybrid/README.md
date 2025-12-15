# LlamaIndex Retrievers Integration: Hybrid

A hybrid retriever that combines multiple retrievers (e.g., dense vector and sparse BM25) using various fusion strategies.

## Installation

```bash
pip install llama-index-retrievers-hybrid
```

For BM25 support:
```bash
pip install llama-index-retrievers-hybrid[bm25]
```

## Overview

Hybrid retrieval combines the strengths of different retrieval methods:
- **Dense (Vector) Retrieval**: Good at capturing semantic similarity
- **Sparse (BM25) Retrieval**: Good at exact keyword matching

By combining both, you get more robust retrieval that handles both semantic queries and keyword-specific searches.

## Fusion Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `RRF` | Reciprocal Rank Fusion | Different score scales, general purpose |
| `RELATIVE_SCORE` | Min-max normalization to [0,1] | When you want interpretable scores |
| `DIST_BASED_SCORE` | Z-score normalization | When score distributions vary significantly |
| `WEIGHTED_SUM` | Direct weighted combination | Similar score scales |

## Usage

### Basic Example

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.retrievers.hybrid import HybridRetriever, FusionMode
from llama_index.retrievers.bm25 import BM25Retriever

# Create your index
documents = [Document(text="...")]
index = VectorStoreIndex.from_documents(documents)

# Create individual retrievers
vector_retriever = index.as_retriever(similarity_top_k=10)
bm25_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=10)

# Create hybrid retriever
hybrid_retriever = HybridRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.6, 0.4],  # 60% vector, 40% BM25
    fusion_mode=FusionMode.RRF,
    similarity_top_k=5,
)

# Retrieve
nodes = hybrid_retriever.retrieve("What is LlamaIndex?")
for node in nodes:
    print(f"Score: {node.score:.4f} - {node.text[:100]}...")
```

### With Query Engine

```python
from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
)

response = query_engine.query("What is LlamaIndex?")
print(response)
```

### Custom Weights

```python
# Emphasize semantic search
hybrid_retriever = HybridRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.8, 0.2],
    fusion_mode=FusionMode.RRF,
)

# Emphasize keyword matching
hybrid_retriever = HybridRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.3, 0.7],
    fusion_mode=FusionMode.RRF,
)
```

### Multiple Retrievers

```python
from llama_index.retrievers.you import YouRetriever

# Combine three retrievers
hybrid_retriever = HybridRetriever(
    retrievers=[vector_retriever, bm25_retriever, you_retriever],
    weights=[0.5, 0.3, 0.2],
    fusion_mode=FusionMode.RRF,
    similarity_top_k=10,
)
```

## Fusion Mode Details

### Reciprocal Rank Fusion (RRF)

```
score = sum(weight_i / (k + rank_i)) for each retriever
```

- Default `k=60`
- Robust to different score scales
- Recommended for most use cases

### Relative Score Fusion

```
normalized_score = (score - min) / (max - min)
final_score = sum(weight_i * normalized_score_i)
```

- Normalizes all scores to [0, 1]
- Good when you need interpretable combined scores

### Distribution-Based Score Fusion

```
z_score = (score - mean) / std
final_score = sum(weight_i * z_score_i)
```

- Uses statistical normalization
- Good when score distributions vary significantly

### Weighted Sum

```
final_score = sum(weight_i * score_i)
```

- Simple direct combination
- Only use when retrievers have similar score scales

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retrievers` | `List[BaseRetriever]` | Required | List of retrievers to combine |
| `weights` | `List[float]` | Equal weights | Weights for each retriever (must sum to 1.0) |
| `fusion_mode` | `FusionMode` | `RRF` | Fusion strategy to use |
| `similarity_top_k` | `int` | `10` | Number of results to return |
| `rrf_k` | `int` | `60` | K parameter for RRF fusion |

## References

- [Reciprocal Rank Fusion Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Hybrid Search in RAG](https://docs.llamaindex.ai/en/stable/examples/retrievers/hybrid_retriever/)
