# LlamaIndex Postprocessor Integration: Heroku Rerank

This package provides a LlamaIndex reranking postprocessor for the Heroku Inference API, enabling you to use Cohere reranking models through Heroku's managed inference service.

## Installation

```bash
pip install llama-index-postprocessor-heroku-rerank
```

## Usage

```python
from llama_index.postprocessor.heroku_rerank import HerokuRerank

# Initialize the reranker
reranker = HerokuRerank(
    api_key="your-heroku-inference-key",
    model="cohere-rerank-3-5",  # default
    top_n=5,
)

# Use with a query engine
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(
    node_postprocessors=[reranker],
    similarity_top_k=20,  # Retrieve 20 documents, rerank to top 5
)

response = query_engine.query("What is the main topic?")
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | str | Required | Your Heroku Inference API key |
| `model` | str | `"cohere-rerank-3-5"` | The reranking model to use |
| `base_url` | str | `"https://us.inference.heroku.com"` | Heroku Inference API base URL |
| `top_n` | int | `5` | Number of top documents to return after reranking |
| `timeout` | float | `60.0` | Request timeout in seconds |

## Available Models

The Heroku Inference API supports the following reranking models:

- `cohere-rerank-3-5` - Latest Cohere reranking model (recommended)
- `cohere-rerank-3` - Previous generation reranking model

## How Reranking Works

1. Your query engine retrieves `similarity_top_k` documents using vector similarity
2. The reranker sends all retrieved documents to the Cohere reranking API
3. Documents are re-scored based on semantic relevance to the query
4. Only the top `top_n` documents are returned for response generation

This two-stage retrieval approach often produces better results than vector similarity alone.

## Example: Complete RAG Pipeline

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.heroku import HerokuEmbedding
from llama_index.llms.heroku import Heroku
from llama_index.postprocessor.heroku_rerank import HerokuRerank
import os

# Initialize Heroku providers
llm = Heroku(model="claude-4-5-sonnet", api_key=os.getenv("INFERENCE_KEY"))
embed_model = HerokuEmbedding(api_key=os.getenv("INFERENCE_KEY"))
reranker = HerokuRerank(api_key=os.getenv("INFERENCE_KEY"), top_n=5)

# Load and index documents
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Query with reranking
query_engine = index.as_query_engine(
    llm=llm,
    node_postprocessors=[reranker],
    similarity_top_k=20,
)

response = query_engine.query("What are the key findings?")
print(response)
```

## License

MIT
