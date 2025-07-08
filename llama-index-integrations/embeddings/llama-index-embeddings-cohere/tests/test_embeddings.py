import os
import tempfile
import httpx
import pytest
from PIL import Image
from llama_index.core.base.embeddings.base import BaseEmbedding

from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.cohere.base import VALID_MODEL_INPUT_TYPES


def test_embedding_class():
    emb = CohereEmbedding(api_key="token")
    assert isinstance(emb, BaseEmbedding)


@pytest.mark.skipif(
    os.environ.get("CO_API_KEY") is None, reason="Cohere API key required"
)
def test_sync_embedding():
    emb = CohereEmbedding(
        api_key=os.environ["CO_API_KEY"],
        model_name="embed-english-v3.0",
        input_type="clustering",
        embedding_type="float",
        httpx_client=httpx.Client(),
    )

    emb.get_query_embedding("I love Cohere!")


@pytest.mark.skipif(
    os.environ.get("CO_API_KEY") is None, reason="Cohere API key required"
)
def test_batch_size_validation():
    """Test that batch size validation works correctly."""
    # Test batch size exceeding the limit
    with pytest.raises(ValueError) as exc_info:
        CohereEmbedding(api_key=os.environ["CO_API_KEY"], embed_batch_size=97)
    assert "exceeds the maximum allowed value of 96" in str(exc_info.value)

    # Test batch size at the limit (should not raise)
    emb = CohereEmbedding(api_key=os.environ["CO_API_KEY"], embed_batch_size=96)
    assert emb.embed_batch_size == 96

    # Test batch size below the limit (should not raise)
    emb = CohereEmbedding(api_key=os.environ["CO_API_KEY"], embed_batch_size=50)
    assert emb.embed_batch_size == 50


@pytest.mark.skipif(
    os.environ.get("CO_API_KEY") is None, reason="Cohere API key required"
)
@pytest.mark.asyncio
async def test_async_embedding():
    emb = CohereEmbedding(
        api_key=os.environ["CO_API_KEY"],
        model_name="embed-english-v3.0",
        input_type="clustering",
        embedding_type="float",
        httpx_async_client=httpx.AsyncClient(),
    )

    await emb.aget_query_embedding("I love Cohere!")


@pytest.mark.skipif(
    os.environ.get("CO_API_KEY") is None, reason="Cohere API key required"
)
@pytest.mark.asyncio
async def test_v4_embedding():
    emb = CohereEmbedding(
        api_key=os.environ["CO_API_KEY"],
        model_name="embed-v4.0",
    )

    embeddings = await emb.aget_text_embedding("I love Cohere!")
    assert len(embeddings) > 0

    embeddings2 = emb.get_text_embedding("I love Cohere!")
    assert len(embeddings2) > 0


@pytest.mark.skipif(
    os.environ.get("CO_API_KEY") is None, reason="Cohere API key required"
)
@pytest.mark.asyncio
async def test_embed_batch():
    emb = CohereEmbedding(
        api_key=os.environ["CO_API_KEY"],
        model_name="embed-v4.0",
    )

    embeddings = await emb.aget_text_embedding_batch(
        ["I love Cohere!", "I love Cohere!"]
    )
    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0
    assert len(embeddings[1]) > 0


@pytest.mark.skipif(
    os.environ.get("CO_API_KEY") is None, reason="Cohere API key required"
)
@pytest.mark.asyncio
async def test_embed_image():
    emb = CohereEmbedding(
        api_key=os.environ["CO_API_KEY"],
        model_name="embed-v4.0",
    )

    # create a test image in a temp file
    image = Image.new("RGB", (100, 100), color="red")
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        image.save(f.name)
        embedding = await emb.aget_image_embedding(f.name)
        embedding2 = emb.get_image_embedding(f.name)

    assert len(embedding) > 0
    assert len(embedding2) > 0


@pytest.mark.skipif(
    os.environ.get("CO_API_KEY") is None, reason="Cohere API key required"
)
@pytest.mark.asyncio
async def test_embed_image_batch():
    emb = CohereEmbedding(
        api_key=os.environ["CO_API_KEY"],
        model_name="embed-v4.0",
    )

    # create a test image in a temp file
    image = Image.new("RGB", (100, 100), color="red")
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        image.save(f.name)
        embeddings = await emb.aget_image_embedding_batch([f.name, f.name])
        embeddings2 = emb.get_image_embedding_batch([f.name, f.name])

    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0
    assert len(embeddings[1]) > 0

    assert len(embeddings2) == 2
    assert len(embeddings2[0]) > 0
    assert len(embeddings2[1]) > 0


@pytest.mark.skipif(
    os.environ.get("CO_API_KEY") is None, reason="Cohere API key required"
)
def test_all_model_names():
    for model_name in VALID_MODEL_INPUT_TYPES:
        emb = CohereEmbedding(
            api_key=os.environ["CO_API_KEY"],
            model_name=model_name,
        )
        embedding = emb.get_text_embedding("Hello, world!")
        assert len(embedding) > 0


def test_cohere_embeddings_custom_endpoint_multiprocessing():
    """
    When used in multiprocessing, the CohereEmbedding instance will be serialized and deserialized. This test
    verifies, that custom base_url's are retained in the spawned processes.
    """
    # Arrange: Create a CohereEmbeddings instance with a custom base_url
    custom_base_url = "test_endpoint"
    api_key = "test_api_key"
    embeddings = CohereEmbedding(api_key=api_key, base_url=custom_base_url)

    # Act: Simulate serialization and deserialization
    serialized_data = embeddings.__getstate__()
    deserialized_embeddings = CohereEmbedding.__new__(CohereEmbedding)
    deserialized_embeddings.__setstate__(serialized_data)

    # Assert: Verify that the deserialized instance retains the correct base_url
    assert deserialized_embeddings.base_url == custom_base_url
