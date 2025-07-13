"""
Google BigQuery Vector Search.

BigQuery Vector Search is a fully managed feature of BigQuery that enables fast,
scalable similarity search over high-dimensional embeddings using approximate
nearest neighbor methods. For more information visit:
https://cloud.google.com/bigquery/docs/vector-search-intro
"""

from llama_index.core.vector_stores.types import BasePydanticVectorStore


class BigQueryVectorStore(BasePydanticVectorStore):
    """BigQuery Vector Store."""
