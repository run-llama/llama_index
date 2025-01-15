import os

import pytest
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatMessage, ImageBlock, MessageRole
from llama_index.llms.gemini import Gemini
from llama_index.llms.gemini.utils import chat_message_to_gemini


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in Gemini.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_chat_message_to_gemini():
    msg = ChatMessage("Some content")
    assert chat_message_to_gemini(msg) == {
        "role": MessageRole.USER,
        "parts": ["Some content"],
    }

    msg = ChatMessage("Some content")
    msg.blocks.append(ImageBlock(image=b"foo", image_mimetype="image/png"))
    assert chat_message_to_gemini(msg) == {
        "role": MessageRole.USER,
        "parts": ["Some content", {"data": b"foo", "mime_type": "image/png"}],
    }


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_generate_image_prompt():
    msg = ChatMessage("Tell me the brand of the car in this image:")
    msg.blocks.append(
        ImageBlock(
            url="https://upload.wikimedia.org/wikipedia/commons/5/52/Ferrari_SP_FFX.jpg"
        )
    )
    response = Gemini().chat(messages=[msg])
    assert "ferrari" in str(response).lower()


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_chat_stream():
    msg = ChatMessage("List three types of software testing strategies")
    response = list(Gemini().stream_chat(messages=[msg]))
    assert response
