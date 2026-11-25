import inspect
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import voyageai
from PIL import Image
from voyageai.video_utils import Video

from llama_index.embeddings.voyageai import VoyageEmbedding


@pytest.mark.parametrize(
    "method",
    [
        "get_query_embedding",
        "aget_query_embedding",
        "get_image_embedding",
        "aget_image_embedding",
        "get_video_embedding",
        "aget_video_embedding",
        "get_video_embeddings",
        "aget_video_embeddings",
    ],
)
@pytest.mark.parametrize(
    ("dimension", "dtype"), [(None, None), (256, None), (None, "int8"), (256, "int8")]
)
@pytest.mark.asyncio
async def test_multimodal_output_options(monkeypatch, method, dimension, dtype):
    response = SimpleNamespace(
        data=[SimpleNamespace(embedding=[1, -2])],
        usage=SimpleNamespace(
            text_tokens=1, image_pixels=0, video_pixels=0, total_tokens=1
        ),
    )
    create = Mock(return_value=response)
    acreate = AsyncMock(return_value=response)
    monkeypatch.setattr(voyageai.MultimodalEmbedding, "create", create)
    monkeypatch.setattr(voyageai.MultimodalEmbedding, "acreate", acreate)
    monkeypatch.setattr(voyageai.Client, "tokenize", Mock(return_value=[[1]]))
    video = Video(
        b"test-video",
        model="voyage-multimodal-3.5",
        optimized=True,
        mime_type="video/mp4",
    )
    monkeypatch.setattr(Video, "from_path", Mock(return_value=video))
    embedding = VoyageEmbedding(
        model_name="voyage-multimodal-3.5",
        voyage_api_key="NOT_A_VALID_KEY",
        output_dimension=dimension,
        output_dtype=dtype,
        truncation=False,
    )

    with BytesIO() as image_file:
        Image.new("RGB", (1, 1)).save(image_file, format="PNG")
        image_file.seek(0)
        if "image" in method:
            argument = image_file
        elif method.endswith("video_embeddings"):
            argument = ["clip.mp4"]
        elif "video" in method:
            argument = "clip.mp4"
        else:
            argument = "a retrieval query"
        result = getattr(embedding, method)(argument)
        if inspect.isawaitable(result):
            result = await result

    assert result == ([[1, -2]] if method.endswith("video_embeddings") else [1, -2])
    request = acreate if method.startswith("a") else create
    other_request = create if method.startswith("a") else acreate
    assert request.call_count == 1
    assert other_request.call_count == 0
    assert request.call_args.kwargs["output_dimension"] == dimension
    assert request.call_args.kwargs["output_dtype"] == dtype
    assert request.call_args.kwargs["truncation"] is False
    assert request.call_args.kwargs["model"] == "voyage-multimodal-3.5"


@pytest.mark.parametrize(
    "method", ["get_text_embedding_batch", "aget_text_embedding_batch"]
)
@pytest.mark.asyncio
async def test_multimodal_text_batch_output_options(monkeypatch, method):
    response = SimpleNamespace(
        data=[SimpleNamespace(embedding=[1, -2]), SimpleNamespace(embedding=[3, -4])],
        usage=SimpleNamespace(
            text_tokens=2, image_pixels=0, video_pixels=0, total_tokens=2
        ),
    )
    create = Mock(return_value=response)
    acreate = AsyncMock(return_value=response)
    monkeypatch.setattr(voyageai.MultimodalEmbedding, "create", create)
    monkeypatch.setattr(voyageai.MultimodalEmbedding, "acreate", acreate)
    monkeypatch.setattr(voyageai.Client, "tokenize", Mock(return_value=[[1]]))
    embedding = VoyageEmbedding(
        model_name="voyage-multimodal-3.5",
        voyage_api_key="NOT_A_VALID_KEY",
        output_dimension=512,
        output_dtype="int8",
    )

    result = getattr(embedding, method)(["first document", "second document"])
    if inspect.isawaitable(result):
        result = await result

    assert result == [[1, -2], [3, -4]]
    request = acreate if method.startswith("a") else create
    assert request.call_count == 1
    assert request.call_args.kwargs["output_dimension"] == 512
    assert request.call_args.kwargs["output_dtype"] == "int8"
    assert request.call_args.kwargs["input_type"] == "document"
    assert request.call_args.kwargs["truncation"] is True
