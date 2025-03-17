import base64
import os

import pytest
import requests
from llama_index.core.llms import LLM
from llama_index.core.schema import ImageNode
from llama_index.multi_modal_llms.gemini import GeminiMultiModal


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in GeminiMultiModal.__mro__]
    assert LLM.__name__ in names_of_base_classes


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
@pytest.mark.asyncio
async def test_streaming_async():
    response = requests.get(
        "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg",
        headers={"User-agent": "Mozilla/5.0"},
    )

    image_str = base64.b64encode(response.content).decode("UTF-8")
    node = ImageNode(image=image_str)

    m = GeminiMultiModal()
    streaming_handler = await m.astream_complete(
        "Tell me what's in this image",
        image_documents=[node],
    )
    async for chunk in streaming_handler:
        assert chunk.delta


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_streaming():
    response = requests.get(
        "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg",
        headers={"User-agent": "Mozilla/5.0"},
    )

    image_str = base64.b64encode(response.content).decode("UTF-8")
    node = ImageNode(image=image_str)

    m = GeminiMultiModal()
    streaming_handler = m.stream_complete(
        "Tell me what's in this image",
        image_documents=[node],
    )
    for chunk in streaming_handler:
        assert chunk.delta
