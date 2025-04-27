import pytest
from llama_index.llms.bedrock_converse.utils import get_model_name
from io import BytesIO
from unittest.mock import MagicMock, patch

from llama_index.core.base.llms.types import (
    AudioBlock,
    ImageBlock,
    MessageRole,
    TextBlock,
)
from llama_index.llms.bedrock_converse.utils import (
    __get_img_format_from_image_mimetype,
    _content_block_to_bedrock_format,
)


def test_get_model_name_translates_us():
    assert (
        get_model_name("us.meta.llama3-2-3b-instruct-v1:0")
        == "meta.llama3-2-3b-instruct-v1:0"
    )


def test_get_model_name_does_nottranslate_cn():
    assert (
        get_model_name("cn.meta.llama3-2-3b-instruct-v1:0")
        == "cn.meta.llama3-2-3b-instruct-v1:0"
    )


def test_get_model_name_does_nottranslate_unsupported():
    assert get_model_name("cohere.command-r-plus-v1:0") == "cohere.command-r-plus-v1:0"


def test_get_model_name_throws_inference_profile_exception():
    with pytest.raises(ValueError):
        assert get_model_name("us.cohere.command-r-plus-v1:0")


def test_get_img_format_jpeg():
    assert __get_img_format_from_image_mimetype("image/jpeg") == "jpeg"


def test_get_img_format_png():
    assert __get_img_format_from_image_mimetype("image/png") == "png"


def test_get_img_format_gif():
    assert __get_img_format_from_image_mimetype("image/gif") == "gif"


def test_get_img_format_webp():
    assert __get_img_format_from_image_mimetype("image/webp") == "webp"


def test_get_img_format_unsupported(caplog):
    result = __get_img_format_from_image_mimetype("image/unsupported")
    assert result == "png"
    assert "Unsupported image mimetype" in caplog.text


def test_content_block_to_bedrock_format_text():
    text_block = TextBlock(text="Hello, world!")
    result = _content_block_to_bedrock_format(text_block, MessageRole.USER)
    assert result == {"text": "Hello, world!"}


@patch("llama_index.core.base.llms.types.ImageBlock.resolve_image")
def test_content_block_to_bedrock_format_image_user(mock_resolve):
    mock_bytes = BytesIO(b"fake_image_data")
    mock_bytes.read = MagicMock(return_value=b"fake_image_data")
    mock_resolve.return_value = mock_bytes

    image_block = ImageBlock(image=b"", image_mimetype="image/png")

    result = _content_block_to_bedrock_format(image_block, MessageRole.USER)

    assert "image" in result
    assert result["image"]["format"] == "png"
    assert "bytes" in result["image"]["source"]
    mock_resolve.assert_called_once()


@patch("llama_index.core.base.llms.types.ImageBlock.resolve_image")
def test_content_block_to_bedrock_format_image_assistant(mock_resolve, caplog):
    image_block = ImageBlock(image=b"", image_mimetype="image/png")
    result = _content_block_to_bedrock_format(image_block, MessageRole.ASSISTANT)

    assert result is None
    assert "only supports image blocks for user messages" in caplog.text
    mock_resolve.assert_not_called()


def test_content_block_to_bedrock_format_audio(caplog):
    audio_block = AudioBlock(audio=b"test_audio")
    result = _content_block_to_bedrock_format(audio_block, MessageRole.USER)

    assert result is None
    assert "Audio blocks are not supported" in caplog.text


def test_content_block_to_bedrock_format_unsupported(caplog):
    unsupported_block = object()
    result = _content_block_to_bedrock_format(unsupported_block, MessageRole.USER)

    assert result is None
    assert "Unsupported block type" in caplog.text
    assert str(type(unsupported_block)) in caplog.text
