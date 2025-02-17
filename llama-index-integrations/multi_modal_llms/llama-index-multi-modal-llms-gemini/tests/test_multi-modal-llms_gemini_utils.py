from io import BytesIO
from unittest.mock import MagicMock

from llama_index.core.schema import ImageDocument
from llama_index.multi_modal_llms.gemini.utils import (
    generate_gemini_multi_modal_chat_message,
)


def test_generate_message_no_image_documents():
    result = generate_gemini_multi_modal_chat_message(
        prompt="Hello", role="user", image_documents=None
    )
    assert result.role == "user"
    assert result.content == "Hello"
    assert result.additional_kwargs == {}


def test_generate_message_empty_image_documents():
    result = generate_gemini_multi_modal_chat_message(
        prompt="Hello", role="user", image_documents=[]
    )
    assert result.role == "user"
    assert result.content == "Hello"
    assert result.additional_kwargs == {}


def test_generate_message_with_image_documents():
    image1 = MagicMock(spec=ImageDocument)
    image1.resolve_image.return_value = BytesIO(b"foo")
    image2 = MagicMock(spec=ImageDocument)
    image2.resolve_image.return_value = BytesIO(b"bar")
    image_documents = [image1, image2]

    result = generate_gemini_multi_modal_chat_message(
        prompt="Hello", role="user", image_documents=image_documents
    )
    assert result.role == "user"
    assert result.content == "Hello"
    assert result.additional_kwargs == {"images": image_documents}
