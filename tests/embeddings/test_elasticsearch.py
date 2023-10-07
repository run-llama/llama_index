import pytest
from llama_index.embeddings.elasticsearch import ElasticsearchEmbeddings

try:
    import elasticsearch
except ImportError:
    elasticsearch = None  # type: ignore


@pytest.fixture()
def model_id() -> str:
    # Replace with your actual model_id
    return "your_model_id"


@pytest.fixture()
def es_url() -> str:
    # Replace with your actual Elasticsearch URL
    return "http://localhost:9200"


@pytest.fixture()
def es_username() -> str:
    # Replace with your actual Elasticsearch username
    return "foo"


@pytest.fixture()
def es_password() -> str:
    # Replace with your actual Elasticsearch password
    return "bar"


@pytest.mark.skipif(elasticsearch is None, reason="elasticsearch not installed")
def test_elasticsearch_embedding_constructor(
    model_id: str, es_url: str, es_username: str, es_password: str
) -> None:
    """Test Elasticsearch embedding query."""
    ElasticsearchEmbeddings.from_credentials(
        model_id=model_id,
        es_url=es_url,
        es_username=es_username,
        es_password=es_password,
    )
