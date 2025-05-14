import unittest
from llama_index.multi_modal_llms.nvidia.utils import (
    infer_image_mimetype_from_base64,
    infer_image_mimetype_from_file_path,
    generate_nvidia_multi_modal_chat_message,
    create_image_content,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
)
from llama_index.core.schema import ImageDocument


class TestFunctions(unittest.TestCase):
    def test_infer_image_mimetype_from_base64(self):
        base64_string = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB"
        self.assertEqual(infer_image_mimetype_from_base64(base64_string), "image/jpeg")

    def test_infer_image_mimetype_from_file_path(self):
        self.assertEqual(infer_image_mimetype_from_file_path("image.jpg"), "jpg")
        self.assertEqual(infer_image_mimetype_from_file_path("image.png"), "png")
        self.assertEqual(infer_image_mimetype_from_file_path("image.webp"), "webp")
        self.assertEqual(infer_image_mimetype_from_file_path("image.gif"), "gif")
        self.assertEqual(infer_image_mimetype_from_file_path("image.txt"), "png")

    # def test_encode_image(self):
    #     image_path = "image.jpg"
    #     encoded_image = encode_image(image_path)
    #     self.assertIsInstance(encoded_image, str)

    def test_create_image_content(self):
        image_document = ImageDocument(image="abcd", mimetype="jpeg")
        content, asset_id = create_image_content(image_document)
        self.assertEqual(content["type"], "text")
        self.assertEqual(content["text"], '<img src="data:image/jpeg;base64,abcd" />')
        self.assertEqual(asset_id, "")

        image_document = ImageDocument(metadata={"asset_id": "12345"}, mimetype="jpeg")
        content, asset_id = create_image_content(image_document)
        self.assertEqual(content["type"], "text")
        self.assertEqual(
            content["text"], '<img src="data:image/jpeg;asset_id,12345" />'
        )
        self.assertEqual(asset_id, "12345")

        image_document = ImageDocument(image_url="https://example.com/image.jpg")
        content, asset_id = create_image_content(image_document)
        self.assertEqual(content["type"], "image_url")
        self.assertEqual(content["image_url"], "https://example.com/image.jpg")
        self.assertEqual(asset_id, "")

    def test_generate_nvidia_multi_modal_chat_message(self):
        inputs = [ChatMessage(role="user", content="Hello")]
        image_documents = [ImageDocument(image="base64_string", mimetype="image/jpeg")]
        message, extra_headers = generate_nvidia_multi_modal_chat_message(
            "google/deplot", inputs=inputs, image_documents=image_documents
        )
        self.assertEqual(len(message[0]), 2)

        inputs = [ChatMessage(role="user", content="Hello")]
        image_documents = [
            ImageDocument(metadata={"asset_id": "12345"}, mimetype="jpeg")
        ]
        message, extra_headers = generate_nvidia_multi_modal_chat_message(
            "google/deplot", inputs=inputs, image_documents=image_documents
        )
        self.assertEqual(len(message[0]), 2)
        self.assertEqual(extra_headers["NVCF-INPUT-ASSET-REFERENCES"], "12345")
