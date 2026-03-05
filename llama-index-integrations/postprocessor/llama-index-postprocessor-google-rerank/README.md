# LlamaIndex Postprocessor Integration: Google Rerank

Uses Google's Discovery Engine Ranking API to rerank search results based on query relevance.

## Installation

```bash
pip install llama-index-postprocessor-google-rerank
```

## Prerequisites

- A Google Cloud project with the [Discovery Engine API](https://console.cloud.google.com/apis/library/discoveryengine.googleapis.com) enabled
- Authentication via [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials) or explicit credentials

## Usage

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.postprocessor.google_rerank import GoogleRerank

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(documents=documents)

reranker = GoogleRerank(
    top_n=3,
    project_id="your-gcp-project-id",
    model="semantic-ranker-default-004",
)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker],
)
response = query_engine.query("What did Sam Altman do in this essay?")
print(response)
```

## Available Models

| Model | Context Window | Notes |
|---|---|---|
| `semantic-ranker-default-004` (default) | 1024 tokens | Latest, multilingual |
| `semantic-ranker-default-003` | 512 tokens | Multilingual |
| `semantic-ranker-default-002` | 512 tokens | English only |

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"semantic-ranker-default-004"` | Ranking model name |
| `top_n` | `int` | `2` | Number of top results to return |
| `project_id` | `str` | `None` | GCP project ID (falls back to `GOOGLE_CLOUD_PROJECT` env var, then ADC) |
| `location` | `str` | `"global"` | GCP location for the ranking config |
| `ranking_config` | `str` | `"default_ranking_config"` | Ranking config resource name |
| `credentials` | `Credentials` | `None` | Optional Google auth credentials object |
