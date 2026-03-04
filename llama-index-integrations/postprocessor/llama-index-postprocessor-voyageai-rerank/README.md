# LlamaIndex Postprocessor Integration: VoyageAI Rerank

This package provides the VoyageAI Rerank integration for LlamaIndex, enabling powerful re-ranking of search results using VoyageAI's state-of-the-art reranker models.

## Installation

```bash
pip install llama-index-postprocessor-voyageai-rerank
```

## Setup

### Get Your API Key

Sign up for a VoyageAI account and obtain your API key from the [VoyageAI Dashboard](https://dash.voyageai.com/).

### Set Environment Variable

```bash
export VOYAGE_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.postprocessor.voyageai_rerank import VoyageAIRerank

# Create documents and index
documents = [
    Document(text="Python is a high-level programming language."),
    Document(text="Machine learning is a branch of artificial intelligence."),
    Document(text="Deep learning uses neural networks with multiple layers."),
]
index = VectorStoreIndex.from_documents(documents)

# Create reranker
reranker = VoyageAIRerank(
    model="rerank-2.5",  # Model to use
    api_key="your-api-key",  # Optional if VOYAGE_API_KEY is set
    top_n=2,  # Return top 2 results
)

# Use with retriever
retriever = index.as_retriever(
    similarity_top_k=5, node_postprocessors=[reranker]
)

nodes = retriever.retrieve("What is machine learning?")
for i, node in enumerate(nodes):
    print(f"{i+1}. Score: {node.score:.4f} - {node.text[:60]}...")
```

### Use with Query Engine

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.postprocessor.voyageai_rerank import VoyageAIRerank

# Setup
documents = [
    Document(text="LlamaIndex is a data framework for LLM applications."),
    Document(text="VoyageAI provides state-of-the-art embedding models."),
    Document(text="Rerankers improve search quality by re-scoring results."),
]
index = VectorStoreIndex.from_documents(documents)

# Create reranker
reranker = VoyageAIRerank(model="rerank-2.5", top_n=3)

# Use with query engine
query_engine = index.as_query_engine(
    similarity_top_k=5, node_postprocessors=[reranker]
)

response = query_engine.query("How do rerankers work?")
print(response)
```

### Combined with VoyageAI Embeddings

```python
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.postprocessor.voyageai_rerank import VoyageAIRerank

# Use VoyageAI for both embeddings and reranking
Settings.embed_model = VoyageEmbedding(model_name="voyage-3.5")

documents = [
    Document(text="Python is a programming language."),
    Document(text="Machine learning uses data to improve performance."),
    Document(text="Neural networks are inspired by the human brain."),
]
index = VectorStoreIndex.from_documents(documents)

# Rerank results
reranker = VoyageAIRerank(model="rerank-2.5", top_n=2)

query_engine = index.as_query_engine(
    similarity_top_k=5, node_postprocessors=[reranker]
)

response = query_engine.query("What is machine learning?")
print(response)
```

## Available Models

VoyageAI offers several reranker models optimized for different use cases:

### Current Models

- **rerank-2.5**: Latest generalist model with 32K context length, instruction-following, and multilingual capabilities (recommended)
- **rerank-2.5-lite**: Optimized for both speed and accuracy, 32K context, multilingual support
- **rerank-2**: Earlier generation model with stable performance
- **rerank-2-lite**: Faster variant of rerank-2

### Legacy Models

- **rerank-1**: Original reranker model
- **rerank-lite-1**: Lightweight variant

For the latest models, see the [VoyageAI Reranker documentation](https://docs.voyageai.com/docs/reranker).

## Configuration Options

| Parameter    | Type           | Default  | Description                                                     |
| ------------ | -------------- | -------- | --------------------------------------------------------------- |
| `model`      | str            | Required | The reranker model to use                                       |
| `api_key`    | str (optional) | None     | VoyageAI API key (falls back to VOYAGE_API_KEY environment var) |
| `top_n`      | int (optional) | None     | Number of top results to return. If None, returns all reranked  |
| `truncation` | bool           | True     | Whether to auto-truncate documents to fit within token limits   |

**Deprecated:**

- `top_k`: Use `top_n` instead

## How Rerankers Work

Rerankers use cross-encoder models to jointly process query-document pairs, providing more accurate relevance scores than embedding-based similarity alone. They work in two stages:

1. **Initial Retrieval**: Vector search retrieves top-k candidates based on embedding similarity
2. **Re-ranking**: The reranker model scores each query-document pair and re-orders results by relevance

This two-stage approach balances speed (fast vector search) with accuracy (precise reranking).

## Features

- **State-of-the-art Models**: Access to VoyageAI's latest reranker models
- **Easy Integration**: Drop-in compatibility with LlamaIndex retrievers and query engines
- **Flexible Configuration**: Control number of results and truncation behavior
- **Multilingual Support**: Works with multiple languages (rerank-2.5 models)
- **32K Context**: Handle long documents with 32,000 token context window
- **Auto-truncation**: Automatically handles documents exceeding token limits

## Context Length Limits

| Model           | Max Query Tokens | Max Document Tokens | Total Context |
| --------------- | ---------------- | ------------------- | ------------- |
| rerank-2.5      | 8,000            | Per document        | 32,000        |
| rerank-2.5-lite | 8,000            | Per document        | 32,000        |
| rerank-2        | 8,000            | Per document        | 4,000         |
| rerank-2-lite   | 8,000            | Per document        | 4,000         |

The reranker can process up to 1,000 documents per request.

## Environment Variables

| Variable         | Description                 |
| ---------------- | --------------------------- |
| `VOYAGE_API_KEY` | VoyageAI API key (required) |

## Best Practices

1. **Use appropriate top_n**: Set `top_n` to limit results to the most relevant documents
2. **Balance initial retrieval**: Retrieve more candidates (e.g., `similarity_top_k=10`) than you need, then use reranker to select the best
3. **Choose the right model**: Use `rerank-2.5` for best quality, `rerank-2.5-lite` for speed
4. **Enable truncation**: Keep `truncation=True` (default) to handle long documents gracefully

## Examples

### Example 1: Basic Reranking

```python
from llama_index.postprocessor.voyageai_rerank import VoyageAIRerank

reranker = VoyageAIRerank(model="rerank-2.5", top_n=3)
```

### Example 2: Without Top-N Filtering

```python
# Return all reranked results
reranker = VoyageAIRerank(model="rerank-2.5")
```

### Example 3: With Custom API Key

```python
reranker = VoyageAIRerank(
    model="rerank-2.5", api_key="your-custom-key", top_n=5
)
```

## Additional Information

For more information about VoyageAI rerankers:

- [VoyageAI Documentation](https://docs.voyageai.com/)
- [VoyageAI Reranker Guide](https://docs.voyageai.com/docs/reranker)
- [VoyageAI Dashboard](https://dash.voyageai.com/)
- [API Reference](https://docs.voyageai.com/reference/reranker-api)

## License

This project is licensed under the MIT License.
