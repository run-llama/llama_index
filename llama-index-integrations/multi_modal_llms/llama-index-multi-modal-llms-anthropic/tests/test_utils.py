import pytest
from unittest.mock import Mock
from llama_index.multi_modal_llms.anthropic.utils import (
    infer_image_mimetype_from_file_path,
    infer_image_mimetype_from_base64,
)


@pytest.fixture()
def sample_base64_png():
    # Create a minimal valid PNG in base64
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="


@pytest.fixture()
def mock_http_response():
    mock_response = Mock()
    mock_response.content = b"fake_image_data"
    return mock_response


def test_infer_image_mimetype_from_base64(sample_base64_png):
    result = infer_image_mimetype_from_base64(sample_base64_png)
    assert result == "image/png"

    # Valid, meaningless base64
    result = infer_image_mimetype_from_base64("lEQVR4nGMAAQAABQABDQ")
    assert result is None


def test_infer_image_mimetype_from_file_path():
    # JPG/JPEG
    assert infer_image_mimetype_from_file_path("image.jpg") == "image/jpeg"
    assert infer_image_mimetype_from_file_path("image.jpeg") == "image/jpeg"

    # PNG
    assert infer_image_mimetype_from_file_path("image.png") == "image/png"

    # GIF
    assert infer_image_mimetype_from_file_path("image.gif") == "image/gif"

    # WEBP
    assert infer_image_mimetype_from_file_path("image.webp") == "image/webp"

    # Catch-all default
    assert infer_image_mimetype_from_file_path("image.asf32") == "image/jpeg"


#
#
# # Tests for generate_anthropic_multi_modal_chat_message
# def test_generate_chat_message_text_only():
#     result = generate_anthropic_multi_modal_chat_message(
#         prompt="Hello, world!",
#         role="user",
#         image_documents=None
#     )
#
#     expected = [{"role": "user", "content": "Hello, world!"}]
#     assert result == expected
#
#
# @patch('llama_index.core.multi_modal_llms.generic_utils.encode_image')
# def test_generate_chat_message_with_image_path(mock_encode_image):
#     mock_encode_image.return_value = "base64_encoded_image"
#
#     image_doc = ImageDocument(
#         image_path="test.jpg",
#         image="",
#         image_url="",
#     )
#
#     result = generate_anthropic_multi_modal_chat_message(
#         prompt="Describe this image",
#         role="user",
#         image_documents=[image_doc]
#     )
#
#     expected = [{
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "source": {
#                     "type": "base64",
#                     "media_type": "image/jpeg",
#                     "data": "base64_encoded_image"
#                 }
#             },
#             {
#                 "type": "text",
#                 "text": "Describe this image"
#             }
#         ]
#     }]
#
#     assert result == expected
#     mock_encode_image.assert_called_once_with("test.jpg")
#
#
# @patch('httpx.get')
# def test_generate_chat_message_with_image_url(mock_get, mock_http_response):
#     mock_get.return_value = mock_http_response
#
#     image_doc = ImageDocument(
#         image_path="",
#         image="",
#         image_url="http://example.com/image.jpg",
#     )
#
#     result = generate_anthropic_multi_modal_chat_message(
#         prompt="Describe this image",
#         role="user",
#         image_documents=[image_doc]
#     )
#
#     expected = [{
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "source": {
#                     "type": "base64",
#                     "media_type": "image/jpeg",
#                     "data": base64.b64encode(b"fake_image_data").decode("utf-8")
#                 }
#             },
#             {
#                 "type": "text",
#                 "text": "Describe this image"
#             }
#         ]
#     }]
#
#     assert result == expected
#     mock_get.assert_called_once_with("http://example.com/image.jpg")
#
#
# def test_generate_chat_message_with_base64_image(sample_base64_image):
#     image_doc = ImageDocument(
#         image_path="",
#         image=sample_base64_image,
#         image_url="",
#     )
#
#     result = generate_anthropic_multi_modal_chat_message(
#         prompt="Describe this image",
#         role="user",
#         image_documents=[image_doc]
#     )
#
#     assert len(result) == 1
#     assert result[0]["role"] == "user"
#     assert len(result[0]["content"]) == 2
#     assert result[0]["content"][0]["type"] == "image"
#     assert result[0]["content"][0]["source"]["type"] == "base64"
#     assert result[0]["content"][0]["source"]["data"] == sample_base64_image
#     assert result[0]["content"][1]["type"] == "text"
#     assert result[0]["content"][1]["text"] == "Describe this image"
#
#
# def test_generate_chat_message_with_metadata_file_path():
#     image_doc = ImageDocument(
#         image_path="",
#         image="",
#         image_url="",
#         metadata={"file_path": "test.png"}
#     )
#
#     with patch('llama_index.core.multi_modal_llms.generic_utils.encode_image') as mock_encode:
#         mock_encode.return_value = "base64_encoded_image"
#
#         result = generate_anthropic_multi_modal_chat_message(
#             prompt="Describe this image",
#             role="user",
#             image_documents=[image_doc]
#         )
#
#         expected = [{
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "source": {
#                         "type": "base64",
#                         "media_type": "image/png",
#                         "data": "base64_encoded_image"
#                     }
#                 },
#                 {
#                     "type": "text",
#                     "text": "Describe this image"
#                 }
#             ]
#         }]
#
#         assert result == expected
#         mock_encode.assert_called_once_with("test.png")
#
#
# # Test edge cases and error handling
# def test_generate_chat_message_empty_image_doc():
#     image_doc = ImageDocument(
#         image_path="",
#         image="",
#         image_url="",
#     )
#
#     result = generate_anthropic_multi_modal_chat_message(
#         prompt="Describe this image",
#         role="user",
#         image_documents=[image_doc]
#     )
#
#     # Should still return a valid message structure even with empty image data
#     assert len(result) == 1
#     assert result[0]["role"] == "user"
#     assert len(result[0]["content"]) == 1  # Only text content
#     assert result[0]["content"][0]["type"] == "text"
#     assert result[0]["content"][0]["text"] == "Describe this image"
#
#
# def test_multiple_image_documents():
#     image_docs = [
#         ImageDocument(image_path="test1.jpg", image="", image_url=""),
#         ImageDocument(image_path="test2.png", image="", image_url=""),
#     ]
#
#     with patch('llama_index.core.multi_modal_llms.generic_utils.encode_image') as mock_encode:
#         mock_encode.return_value = "base64_encoded_image"
#
#         result = generate_anthropic_multi_modal_chat_message(
#             prompt="Describe these images",
#             role="user",
#             image_documents=image_docs
#         )
#
#         assert len(result) == 1
#         assert len(result[0]["content"]) == 3  # 2 images + 1 text
#         assert result[0]["content"][0]["type"] == "image"
#         assert result[0]["content"][1]["type"] == "image"
#         assert result[0]["content"][2]["type"] == "text"
#         assert mock_encode.call_count == 2
