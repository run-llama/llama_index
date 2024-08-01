import pytest
from llama_index.embeddings.llm_rails import LLMRailsEmbedding


@pytest.fixture()
def model_id() -> str:
    # Replace with model name
    return "your_model_id"


@pytest.fixture()
def api_key() -> str:
    # Replace with your api key
    return "your_api_key"


def test_llm_rails_embedding_constructor(model_id: str, api_key: str) -> None:
    """Test LLMRails embedding constructor."""
    LLMRailsEmbedding(model_id=model_id, api_key=api_key)
