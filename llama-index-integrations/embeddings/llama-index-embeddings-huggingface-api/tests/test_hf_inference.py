from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from llama_index.embeddings.huggingface_api.base import HuggingFaceInferenceAPIEmbedding
from llama_index.embeddings.huggingface_api.pooling import Pooling

STUB_MODEL_NAME = "placeholder_model"


@pytest.fixture(name="hf_inference_api_embedding")
def fixture_hf_inference_api_embedding() -> HuggingFaceInferenceAPIEmbedding:
    with patch.dict("sys.modules", huggingface_hub=MagicMock()):
        return HuggingFaceInferenceAPIEmbedding(model_name=STUB_MODEL_NAME)


class TestHuggingFaceInferenceAPIEmbeddings:
    def test_class_name(
        self, hf_inference_api_embedding: HuggingFaceInferenceAPIEmbedding
    ) -> None:
        assert (
            HuggingFaceInferenceAPIEmbedding.class_name()
            == HuggingFaceInferenceAPIEmbedding.__name__
        )
        assert (
            hf_inference_api_embedding.class_name()
            == HuggingFaceInferenceAPIEmbedding.__name__
        )

    # def test_using_recommended_model(self) -> None:
    #     mock_hub = MagicMock()
    #     mock_hub.InferenceClient.get_recommended_model.return_value = (
    #         "facebook/bart-base"
    #     )
    #     with patch.dict("sys.modules", huggingface_hub=mock_hub):
    #         embedding = HuggingFaceInferenceAPIEmbedding(task="feature-extraction")
    #     assert embedding.model_name == "facebook/bart-base"
    #     # mock_hub.InferenceClient.get_recommended_model.assert_called_once_with(
    #     #     task="feature-extraction"
    #     # )

    def test_embed_query(
        self, hf_inference_api_embedding: HuggingFaceInferenceAPIEmbedding
    ) -> None:
        raw_single_embedding = np.random.default_rng().random(
            (1, 3, 1024), dtype=np.float32
        )

        hf_inference_api_embedding.pooling = Pooling.CLS
        with patch.object(
            hf_inference_api_embedding._async_client,
            "feature_extraction",
            AsyncMock(return_value=raw_single_embedding),
        ) as mock_feature_extraction:
            embedding = hf_inference_api_embedding.get_query_embedding("test")
        assert isinstance(embedding, list)
        assert len(embedding) == 1024
        assert isinstance(embedding[0], float)
        assert np.all(
            np.array(embedding, dtype=raw_single_embedding.dtype)
            == raw_single_embedding[0, 0]
        )
        mock_feature_extraction.assert_awaited_once_with("test")

        hf_inference_api_embedding.pooling = Pooling.MEAN
        with patch.object(
            hf_inference_api_embedding._async_client,
            "feature_extraction",
            AsyncMock(return_value=raw_single_embedding),
        ) as mock_feature_extraction:
            embedding = hf_inference_api_embedding.get_query_embedding("test")
        assert isinstance(embedding, list)
        assert len(embedding) == 1024
        assert isinstance(embedding[0], float)
        assert np.all(
            np.array(embedding, dtype=raw_single_embedding.dtype)
            == raw_single_embedding[0].mean(axis=0)
        )
        mock_feature_extraction.assert_awaited_once_with("test")

    def test_embed_query_one_dimension(
        self, hf_inference_api_embedding: HuggingFaceInferenceAPIEmbedding
    ) -> None:
        raw_single_embedding = np.random.default_rng().random(1024, dtype=np.float32)

        with patch.object(
            hf_inference_api_embedding._async_client,
            "feature_extraction",
            AsyncMock(return_value=raw_single_embedding),
        ) as mock_feature_extraction:
            embedding = hf_inference_api_embedding.get_query_embedding("test")
        assert isinstance(embedding, list)
        assert len(embedding) == 1024
        assert isinstance(embedding[0], float)
        assert np.all(
            np.array(embedding, dtype=raw_single_embedding.dtype)
            == raw_single_embedding
        )
        mock_feature_extraction.assert_awaited_once_with("test")

    def test_serialization(
        self, hf_inference_api_embedding: HuggingFaceInferenceAPIEmbedding
    ) -> None:
        serialized = hf_inference_api_embedding.to_dict()
        # Check Hugging Face Inference API base class specifics
        assert serialized["model_name"] == STUB_MODEL_NAME
        # Check Hugging Face Inference API Embeddings derived class specifics
        assert serialized["pooling"] == Pooling.CLS

    def test_serde(
        self, hf_inference_api_embedding: HuggingFaceInferenceAPIEmbedding
    ) -> None:
        serialized = hf_inference_api_embedding.model_dump()
        deserialized = HuggingFaceInferenceAPIEmbedding.model_validate(serialized)
        assert deserialized.headers == hf_inference_api_embedding.headers
