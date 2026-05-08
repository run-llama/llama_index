import os
import tempfile
from io import BytesIO
from unittest.mock import MagicMock, AsyncMock, patch

import cohere
import httpx
import pytest
from PIL import Image
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.types import AudioBlock, ImageBlock, TextBlock

from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.cohere.base import (
    VALID_MODEL_INPUT_TYPES,
    _create_retry_decorator,
)


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


def test_create_retry_decorator():
    """Test that _create_retry_decorator creates a working decorator."""
    decorator = _create_retry_decorator(max_retries=3)
    assert decorator is not None

    call_count = 0

    @decorator
    def failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise cohere.errors.ServiceUnavailableError(body=None)
        return "success"

    result = failing_function()
    assert result == "success"
    assert call_count == 3


def test_create_retry_decorator_retries_on_internal_server_error():
    """Test that retry decorator retries on InternalServerError."""
    decorator = _create_retry_decorator(max_retries=3)
    call_count = 0

    @decorator
    def failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise cohere.errors.InternalServerError(body=None)
        return "success"

    result = failing_function()
    assert result == "success"
    assert call_count == 2


def test_create_retry_decorator_retries_on_gateway_timeout():
    """Test that retry decorator retries on GatewayTimeoutError."""
    decorator = _create_retry_decorator(max_retries=3)
    call_count = 0

    @decorator
    def failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise cohere.errors.GatewayTimeoutError(body=None)
        return "success"

    result = failing_function()
    assert result == "success"
    assert call_count == 2


def test_embed_with_retry():
    """Test that _embed uses retry logic."""
    emb = CohereEmbedding(api_key="test_key", max_retries=3)

    mock_embeddings = MagicMock()
    mock_embeddings.float = [[0.1, 0.2, 0.3]]

    mock_response = MagicMock()
    mock_response.embeddings = mock_embeddings

    mock_client = MagicMock()
    mock_client.embed.return_value = mock_response

    with patch.object(emb, "_get_client", return_value=mock_client):
        result = emb._embed(texts=["test text"])

    assert result == [[0.1, 0.2, 0.3]]
    mock_client.embed.assert_called_once()


@pytest.mark.asyncio
async def test_aembed_with_retry():
    """Test that _aembed uses retry logic."""
    emb = CohereEmbedding(api_key="test_key", max_retries=3)

    mock_embeddings = MagicMock()
    mock_embeddings.float = [[0.1, 0.2, 0.3]]

    mock_response = MagicMock()
    mock_response.embeddings = mock_embeddings

    mock_client = MagicMock()
    mock_client.embed = AsyncMock(return_value=mock_response)

    with patch.object(emb, "_get_async_client", return_value=mock_client):
        result = await emb._aembed(texts=["test text"])

    assert result == [[0.1, 0.2, 0.3]]
    mock_client.embed.assert_called_once()


def test_embed_image_with_retry():
    """Test that _embed_image uses retry logic."""
    emb = CohereEmbedding(api_key="test_key", model_name="embed-v4.0", max_retries=3)

    mock_embeddings = MagicMock()
    mock_embeddings.float = [[0.1, 0.2, 0.3]]

    mock_response = MagicMock()
    mock_response.embeddings = mock_embeddings

    mock_client = MagicMock()
    mock_client.embed.return_value = mock_response

    with (
        patch.object(emb, "_get_client", return_value=mock_client),
        patch.object(
            emb,
            "_image_to_base64_data_url",
            return_value="data:image/png;base64,test",
        ),
    ):
        result = emb._embed_image(image_paths=["test.png"], input_type="image")

    assert result == [[0.1, 0.2, 0.3]]
    mock_client.embed.assert_called_once()


@pytest.mark.asyncio
async def test_aembed_image_with_retry():
    """Test that _aembed_image uses retry logic."""
    emb = CohereEmbedding(api_key="test_key", model_name="embed-v4.0", max_retries=3)

    mock_embeddings = MagicMock()
    mock_embeddings.float = [[0.1, 0.2, 0.3]]

    mock_response = MagicMock()
    mock_response.embeddings = mock_embeddings

    mock_client = MagicMock()
    mock_client.embed = AsyncMock(return_value=mock_response)

    with (
        patch.object(emb, "_get_async_client", return_value=mock_client),
        patch.object(
            emb,
            "_image_to_base64_data_url",
            return_value="data:image/png;base64,test",
        ),
    ):
        result = await emb._aembed_image(image_paths=["test.png"], input_type="image")

    assert result == [[0.1, 0.2, 0.3]]
    mock_client.embed.assert_called_once()


def test_max_retries_parameter():
    """Test that max_retries parameter is properly set."""
    emb = CohereEmbedding(api_key="test_key", max_retries=5)
    assert emb.max_retries == 5

    emb_default = CohereEmbedding(api_key="test_key")
    assert emb_default.max_retries == 10


def _tiny_png_bytes() -> bytes:
    img = Image.new("RGB", (1, 1), color=(255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_supports_mixed_embedding_cohere():
    emb_v4 = CohereEmbedding(api_key="k", model_name="embed-v4.0")
    assert emb_v4.supports_mixed_embedding is True
    emb_v3 = CohereEmbedding(api_key="k", model_name="embed-english-v3.0")
    assert emb_v3.supports_mixed_embedding is True
    emb_v2 = CohereEmbedding(api_key="k", model_name="embed-english-v2.0")
    assert emb_v2.supports_mixed_embedding is False


def test_mixed_content_cohere_supported_and_blocks_to_api_format():
    png = _tiny_png_bytes()
    content = [
        TextBlock(text="caption"),
        ImageBlock(image=png),
        AudioBlock(audio=b"x"),
    ]
    supported = CohereEmbedding._mixed_content_cohere_supported(content)
    assert len(supported) == 2
    api = CohereEmbedding._blocks_to_cohere_api_format(supported)
    assert api[0] == {"type": "text", "text": "caption"}
    assert api[1]["type"] == "image_url"
    assert "url" in api[1]["image_url"]


def test_get_mixed_content_embedding_cohere():
    emb = CohereEmbedding(api_key="test_key", model_name="embed-v4.0", max_retries=1)
    mock_embeddings = MagicMock()
    mock_embeddings.float = [[0.1, 0.2, 0.3]]
    mock_response = MagicMock()
    mock_response.embeddings = mock_embeddings
    mock_client = MagicMock()
    mock_client.embed.return_value = mock_response
    blocks = [TextBlock(text="hi"), ImageBlock(image=_tiny_png_bytes())]
    with patch.object(emb, "_get_client", return_value=mock_client):
        out = emb._get_mixed_content_embedding(blocks)
    assert out == [0.1, 0.2, 0.3]
    mock_client.embed.assert_called_once()
    call_kw = mock_client.embed.call_args.kwargs
    assert "inputs" in call_kw
    assert call_kw["inputs"][0]["content"][0]["type"] == "text"


@pytest.mark.asyncio
async def test_aget_mixed_content_embedding_cohere():
    emb = CohereEmbedding(api_key="test_key", model_name="embed-v4.0", max_retries=1)
    mock_embeddings = MagicMock()
    mock_embeddings.float = [[0.4, 0.5]]
    mock_response = MagicMock()
    mock_response.embeddings = mock_embeddings
    mock_client = MagicMock()
    mock_client.embed = AsyncMock(return_value=mock_response)
    blocks = [TextBlock(text="hi"), ImageBlock(image=_tiny_png_bytes())]
    with patch.object(emb, "_get_async_client", return_value=mock_client):
        out = await emb._aget_mixed_content_embedding(blocks)
    assert out == [0.4, 0.5]


def test_get_mixed_content_embeddings_cohere_batch():
    emb = CohereEmbedding(api_key="test_key", model_name="embed-v4.0", max_retries=1)
    mock_embeddings = MagicMock()
    mock_embeddings.float = [[0.1], [0.2]]
    mock_response = MagicMock()
    mock_response.embeddings = mock_embeddings
    mock_client = MagicMock()
    mock_client.embed.return_value = mock_response
    png = _tiny_png_bytes()
    contents = [
        [TextBlock(text="a"), ImageBlock(image=png)],
        [TextBlock(text="b"), ImageBlock(image=png)],
    ]
    with patch.object(emb, "_get_client", return_value=mock_client):
        out = emb._get_mixed_content_embeddings(contents)
    assert out == [[0.1], [0.2]]


@pytest.mark.asyncio
async def test_aget_mixed_content_embeddings_cohere_batch():
    emb = CohereEmbedding(api_key="test_key", model_name="embed-v4.0", max_retries=1)
    mock_embeddings = MagicMock()
    mock_embeddings.float = [[0.3], [0.4]]
    mock_response = MagicMock()
    mock_response.embeddings = mock_embeddings
    mock_client = MagicMock()
    mock_client.embed = AsyncMock(return_value=mock_response)
    png = _tiny_png_bytes()
    contents = [
        [TextBlock(text="a"), ImageBlock(image=png)],
        [TextBlock(text="b"), ImageBlock(image=png)],
    ]
    with patch.object(emb, "_get_async_client", return_value=mock_client):
        out = await emb._aget_mixed_content_embeddings(contents)
    assert out == [[0.3], [0.4]]


def test_get_mixed_content_embeddings_cohere_empty_list():
    emb = CohereEmbedding(api_key="test_key", model_name="embed-v4.0")
    assert emb._get_mixed_content_embeddings([]) == []


@pytest.mark.asyncio
async def test_aget_mixed_content_embeddings_cohere_empty_list():
    emb = CohereEmbedding(api_key="test_key", model_name="embed-v4.0")
    assert await emb._aget_mixed_content_embeddings([]) == []


def test_get_mixed_content_embedding_cohere_no_supported_blocks_raises():
    emb = CohereEmbedding(api_key="k", model_name="embed-v4.0")
    with pytest.raises(ValueError, match="Cohere-supported"):
        emb._get_mixed_content_embedding([AudioBlock(audio=b"x")])


def test_get_mixed_content_embedding_cohere_wrong_model_raises():
    emb = CohereEmbedding(api_key="k", model_name="embed-english-v2.0")
    png = _tiny_png_bytes()
    with pytest.raises(ValueError, match="not a valid multi-modal"):
        emb._get_mixed_content_embedding([TextBlock(text="a"), ImageBlock(image=png)])


def test_get_mixed_content_embeddings_cohere_batch_empty_after_filter_raises():
    emb = CohereEmbedding(api_key="k", model_name="embed-v4.0")
    with pytest.raises(ValueError, match="Every mixed content item"):
        emb._get_mixed_content_embeddings(
            [
                [TextBlock(text="a"), ImageBlock(image=_tiny_png_bytes())],
                [AudioBlock(audio=b"x")],
            ]
        )


@pytest.mark.asyncio
async def test_aget_mixed_content_embeddings_cohere_batch_empty_after_filter_raises():
    emb = CohereEmbedding(api_key="k", model_name="embed-v4.0")
    with pytest.raises(ValueError, match="Every mixed content item"):
        await emb._aget_mixed_content_embeddings(
            [
                [TextBlock(text="a"), ImageBlock(image=_tiny_png_bytes())],
                [AudioBlock(audio=b"x")],
            ]
        )
