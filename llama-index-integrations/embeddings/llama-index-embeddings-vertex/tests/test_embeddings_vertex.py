import io
import unittest
from unittest.mock import patch, Mock, MagicMock, AsyncMock

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings import MultiModalEmbedding
from vertexai.language_models import TextEmbedding, TextEmbeddingInput
from vertexai.vision_models import MultiModalEmbeddingResponse

from PIL import Image as PillowImage

from llama_index.embeddings.vertex import (
    VertexTextEmbedding,
    VertexMultiModalEmbedding,
    VertexEmbeddingMode,
)
from llama_index.embeddings.vertex.base import (
    _get_embedding_request,
    _UNSUPPORTED_TASK_TYPE_MODEL,
)


class VertexTextEmbeddingTest(unittest.TestCase):
    @patch("vertexai.init")
    @patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
    def test_init(self, model_mock: Mock, mock_init: Mock):
        mock_cred = Mock(return_value="mock_credentials_instance")
        embedding = VertexTextEmbedding(
            model_name="textembedding-gecko@001",
            project="test-project",
            location="us-test-location",
            credentials=mock_cred,
            embed_mode=VertexEmbeddingMode.RETRIEVAL_MODE,
            embed_batch_size=100,
            num_workers=2,
        )

        mock_init.assert_called_once_with(
            project="test-project",
            location="us-test-location",
            credentials=mock_cred,
        )

        self.assertIsInstance(embedding, BaseEmbedding)

        self.assertEqual(embedding.model_name, "textembedding-gecko@001")
        self.assertEqual(embedding.embed_mode, VertexEmbeddingMode.RETRIEVAL_MODE)
        self.assertEqual(embedding.embed_batch_size, 100)
        self.assertEqual(embedding.num_workers, 2)

    @patch("vertexai.init")
    @patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
    def test_get_embedding_retrieval(self, model_mock: Mock, init_mock: Mock):
        model = MagicMock()
        model_mock.return_value = model
        mock_cred = Mock(return_value="mock_credentials_instance")
        embedding = VertexTextEmbedding(
            project="test-project",
            location="us-test-location",
            credentials=mock_cred,
            embed_mode=VertexEmbeddingMode.RETRIEVAL_MODE,
            additional_kwargs={"auto_truncate": True},
        )

        model.get_embeddings.return_value = [TextEmbedding(values=[0.1, 0.2, 0.3])]
        result = embedding.get_text_embedding("some text")

        model.get_embeddings.assert_called_once()
        positional_args, keyword_args = model.get_embeddings.call_args
        model.get_embeddings.reset_mock()

        self.assertEqual(len(positional_args[0]), 1)
        self.assertEqual(positional_args[0][0].text, "some text")
        self.assertEqual(positional_args[0][0].task_type, "RETRIEVAL_DOCUMENT")
        self.assertEqual(result, [0.1, 0.2, 0.3])
        self.assertTrue(keyword_args["auto_truncate"])

        model.get_embeddings.return_value = [TextEmbedding(values=[0.1, 0.2, 0.3])]
        result = embedding.get_query_embedding("some query text")

        model.get_embeddings.assert_called_once()
        positional_args, keyword_args = model.get_embeddings.call_args

        self.assertEqual(len(positional_args[0]), 1)
        self.assertEqual(positional_args[0][0].text, "some query text")
        self.assertEqual(positional_args[0][0].task_type, "RETRIEVAL_QUERY")
        self.assertEqual(result, [0.1, 0.2, 0.3])
        self.assertTrue(keyword_args["auto_truncate"])

    def test_unsupported_task_type_model(self):
        texts = ["text1", "text2"]
        for model_name in _UNSUPPORTED_TASK_TYPE_MODEL:
            with self.subTest(model_name=model_name):
                result = _get_embedding_request(
                    texts, VertexEmbeddingMode.RETRIEVAL_MODE, False, model_name
                )
                self.assertTrue(
                    all(isinstance(item, TextEmbeddingInput) for item in result)
                )
                self.assertTrue(all(item.task_type is None for item in result))

    def test_supported_task_type_model(self):
        texts = ["text1", "text2"]
        model_name = "textembedding-gecko@003"
        result = _get_embedding_request(
            texts, VertexEmbeddingMode.RETRIEVAL_MODE, False, model_name
        )

        self.assertTrue(all(isinstance(item, TextEmbeddingInput) for item in result))
        self.assertTrue(all(item.task_type == "RETRIEVAL_DOCUMENT" for item in result))


class VertexTextEmbeddingTestAsync(unittest.IsolatedAsyncioTestCase):
    @patch("vertexai.init")
    @patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
    async def test_get_embedding_retrieval(
        self, model_mock: AsyncMock, init_mock: AsyncMock
    ):
        model = MagicMock()
        model.get_embeddings_async = (
            AsyncMock()
        )  # Ensure get_embeddings is an AsyncMock for async calls
        model_mock.return_value = model
        mock_cred = Mock(return_value="mock_credentials_instance")

        embedding = VertexTextEmbedding(
            project="test-project",
            location="us-test-location",
            embed_mode=VertexEmbeddingMode.RETRIEVAL_MODE,
            additional_kwargs={"auto_truncate": True},
            credentials=mock_cred,
        )

        model.get_embeddings_async.return_value = [
            TextEmbedding(values=[0.1, 0.2, 0.3])
        ]
        result = await embedding.aget_text_embedding("some text")

        model.get_embeddings_async.assert_called_once()
        positional_args, keyword_args = model.get_embeddings_async.call_args
        model.get_embeddings_async.reset_mock()

        self.assertEqual(len(positional_args[0]), 1)
        self.assertEqual(positional_args[0][0].text, "some text")
        self.assertEqual(positional_args[0][0].task_type, "RETRIEVAL_DOCUMENT")
        self.assertEqual(result, [0.1, 0.2, 0.3])
        self.assertTrue(keyword_args["auto_truncate"])

        model.get_embeddings_async.return_value = [
            TextEmbedding(values=[0.1, 0.2, 0.3])
        ]
        result = await embedding.aget_query_embedding("some query text")

        model.get_embeddings_async.assert_called_once()
        positional_args, keyword_args = model.get_embeddings_async.call_args

        self.assertEqual(len(positional_args[0]), 1)
        self.assertEqual(positional_args[0][0].text, "some query text")
        self.assertEqual(positional_args[0][0].task_type, "RETRIEVAL_QUERY")
        self.assertEqual(result, [0.1, 0.2, 0.3])
        self.assertTrue(keyword_args["auto_truncate"])


class VertexMultiModalEmbeddingTest(unittest.TestCase):
    @patch("vertexai.init")
    @patch("vertexai.vision_models.MultiModalEmbeddingModel.from_pretrained")
    def test_init(self, model_mock: Mock, mock_init: Mock):
        mock_cred = Mock(return_value="mock_credentials_instance")
        embedding = VertexMultiModalEmbedding(
            model_name="multimodalembedding",
            project="test-project",
            location="us-test-location",
            credentials=mock_cred,
            embed_dimension=1408,
            embed_batch_size=100,
        )

        mock_init.assert_called_once_with(
            project="test-project",
            location="us-test-location",
            credentials=mock_cred,
        )

        self.assertIsInstance(embedding, MultiModalEmbedding)

        self.assertEqual(embedding.model_name, "multimodalembedding")
        self.assertEqual(embedding.embed_batch_size, 100)
        self.assertEqual(embedding.embed_dimension, 1408)

    @patch("vertexai.init")
    @patch("vertexai.vision_models.MultiModalEmbeddingModel.from_pretrained")
    def test_text_embedding(self, model_mock: Mock, init_mock: Mock):
        model = MagicMock()
        model_mock.return_value = model

        embedding = VertexMultiModalEmbedding(
            project="test-project",
            location="us-test-location",
            embed_dimension=1408,
            additional_kwargs={"additional_kwarg": True},
        )

        model.get_embeddings.return_value = MultiModalEmbeddingResponse(
            _prediction_response=None, text_embedding=[0.1, 0.2, 0.3]
        )

        result = embedding.get_text_embedding("some text")
        self.assertEqual(result, [0.1, 0.2, 0.3])

        model.get_embeddings.assert_called_once()
        positional_args, keyword_args = model.get_embeddings.call_args

        self.assertEqual(keyword_args["contextual_text"], "some text")
        self.assertEqual(keyword_args["dimension"], 1408)
        self.assertTrue(keyword_args["additional_kwarg"])

    @patch("vertexai.init")
    @patch("vertexai.vision_models.Image.load_from_file")
    @patch("vertexai.vision_models.MultiModalEmbeddingModel.from_pretrained")
    def test_image_embedding_path(
        self, model_mock: Mock, load_file_mock: Mock, init_mock: Mock
    ):
        model = MagicMock()
        model_mock.return_value = model

        embedding = VertexMultiModalEmbedding(
            project="test-project",
            location="us-test-location",
            embed_dimension=1408,
            additional_kwargs={"additional_kwarg": True},
        )

        model.get_embeddings.return_value = MultiModalEmbeddingResponse(
            _prediction_response=None, image_embedding=[0.1, 0.2, 0.3]
        )

        result = embedding.get_image_embedding("data/test-image.jpg")
        self.assertEqual(result, [0.1, 0.2, 0.3])

        model.get_embeddings.assert_called_once()
        positional_args, keyword_args = model.get_embeddings.call_args

        load_file_mock.assert_called_once_with("data/test-image.jpg")
        self.assertTrue("image" in keyword_args)
        self.assertEqual(keyword_args["dimension"], 1408)
        self.assertTrue(keyword_args["additional_kwarg"])

    @patch("vertexai.init")
    @patch("vertexai.vision_models.Image.load_from_file")
    @patch("vertexai.vision_models.MultiModalEmbeddingModel.from_pretrained")
    def test_image_embedding_bytes(
        self, model_mock: Mock, load_file_mock: Mock, init_mock: Mock
    ):
        model = MagicMock()
        model_mock.return_value = model

        embedding = VertexMultiModalEmbedding(
            project="test-project",
            location="us-test-location",
            embed_dimension=1408,
            additional_kwargs={"additional_kwarg": True},
        )

        model.get_embeddings.return_value = MultiModalEmbeddingResponse(
            _prediction_response=None, image_embedding=[0.1, 0.2, 0.3]
        )

        image = PillowImage.new("RGB", (128, 128))
        bytes_io = io.BytesIO()
        image.save(bytes_io, "jpeg")
        bytes_io.seek(0)

        result = embedding.get_image_embedding(bytes_io)
        self.assertEqual(result, [0.1, 0.2, 0.3])

        model.get_embeddings.assert_called_once()
        positional_args, keyword_args = model.get_embeddings.call_args

        load_file_mock.assert_not_called()
        self.assertEqual(keyword_args["dimension"], 1408)
        self.assertTrue(keyword_args["additional_kwarg"])

    def test_schema(self):
        schema = VertexTextEmbedding.model_json_schema()

        self.assertIsInstance(schema, dict)
        self.assertEqual(schema["properties"]["embed_mode"].get("default"), "retrieval")
        self.assertEqual(schema["properties"]["embed_batch_size"].get("default"), 10)
        self.assertEqual(schema["properties"]["num_workers"].get("default"), None)
        self.assertEqual(schema["properties"]["client_email"].get("default"), None)
        self.assertEqual(schema["properties"]["token_uri"].get("default"), None)
        self.assertEqual(schema["properties"]["private_key_id"].get("default"), None)
        self.assertEqual(schema["properties"]["private_key"].get("default"), None)

        self.assertEqual(schema.get("required", []), [])


if __name__ == "__main__":
    unittest.main()
