import pytest

from llama_index.embeddings.nvidia import NVIDIAEmbedding


@pytest.mark.integration()
def test_basic(model: str, mode: dict) -> None:
    client = NVIDIAEmbedding(model=model, **mode)
    response = client.get_query_embedding("Hello, world!")
    assert isinstance(response, list)
    assert len(response) > 0
    assert isinstance(response[0], float)


## ================== nvidia/llama-3.2-nv-embedqa-1b-v2 model dimensions param test cases ==================
@pytest.mark.integration()
@pytest.mark.parametrize("dimensions", [32, 64, 128, 2048])
def test_embed_text_with_dimensions(mode: dict, dimensions: int) -> None:
    model = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    query = "foo bar"
    embedding = NVIDIAEmbedding(model=model, dimensions=dimensions)
    assert len(embedding.get_query_embedding(query)) == dimensions


@pytest.mark.integration()
@pytest.mark.parametrize("dimensions", [32, 64, 128, 2048])
def test_embed_query_with_dimensions(dimensions: int) -> None:
    model = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    query = "foo bar"
    embedding = NVIDIAEmbedding(model=model, dimensions=dimensions)
    assert len(embedding.get_query_embedding(query)) == dimensions


@pytest.mark.integration()
@pytest.mark.parametrize("dimensions", [102400])
def test_embed_query_with_large_dimensions(dimensions: int) -> None:
    model = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    query = "foo bar"
    embedding = NVIDIAEmbedding(model=model, dimensions=dimensions)
    assert 2048 <= len(embedding.get_query_embedding(query)) < dimensions


@pytest.mark.integration()
@pytest.mark.parametrize("dimensions", [102400])
def test_embed_documents_with_large_dimensions(dimensions: int) -> None:
    model = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    documents = ["foo bar", "bar foo"]
    embedding = NVIDIAEmbedding(model=model, dimensions=dimensions)
    output = embedding.get_text_embedding_batch(documents)
    assert len(output) == len(documents)
    assert all(2048 <= len(doc) < dimensions for doc in output)


@pytest.mark.integration()
@pytest.mark.parametrize("dimensions", [-1])
def test_embed_query_invalid_dimensions(dimensions: int) -> None:
    model = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    query = "foo bar"
    with pytest.raises(Exception) as exc:
        NVIDIAEmbedding(model=model, dimensions=dimensions).get_query_embedding(query)
    assert "400" in str(exc.value)


@pytest.mark.integration()
@pytest.mark.parametrize("dimensions", [-1])
def test_embed_documents_invalid_dimensions(dimensions: int) -> None:
    model = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    documents = ["foo bar", "bar foo"]
    with pytest.raises(Exception) as exc:
        NVIDIAEmbedding(model=model, dimensions=dimensions).get_text_embedding_batch(
            documents
        )
    assert "400" in str(exc.value)
