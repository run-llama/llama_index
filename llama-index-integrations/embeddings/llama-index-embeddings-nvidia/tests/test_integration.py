import pytest

from llama_index.embeddings.nvidia import NVIDIAEmbedding


@pytest.mark.integration()
def test_basic(model: str, mode: dict) -> None:
    client = NVIDIAEmbedding(model=model, **mode)
    response = client.get_query_embedding("Hello, world!")
    assert isinstance(response, list)
    assert len(response) > 0
    assert isinstance(response[0], float)
