from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from llama_index.embeddings.pooling import Pooling

from tests.llms.test_huggingface import STUB_MODEL_NAME


@pytest.fixture(name="hf_inference_api_embeddings")
def fixture_hf_inference_api_embeddings() -> HuggingFaceInferenceAPIEmbeddings:
    with patch.dict("sys.modules", huggingface_hub=MagicMock()):
        return HuggingFaceInferenceAPIEmbeddings(model_name=STUB_MODEL_NAME)


class TestHuggingFaceInferenceAPIEmbeddings:
    def test_class_name(
        self, hf_inference_api_embeddings: HuggingFaceInferenceAPIEmbeddings
    ) -> None:
        assert (
            HuggingFaceInferenceAPIEmbeddings.class_name()
            == HuggingFaceInferenceAPIEmbeddings.__name__
        )
        assert (
            hf_inference_api_embeddings.class_name()
            == HuggingFaceInferenceAPIEmbeddings.__name__
        )

    def test_embed_query(
        self, hf_inference_api_embeddings: HuggingFaceInferenceAPIEmbeddings
    ) -> None:
        raw_single_embedding = np.random.rand(1, 3, 1024)

        hf_inference_api_embeddings.pooling = Pooling.CLS
        with patch.object(
            hf_inference_api_embeddings._async_client,
            "feature_extraction",
            AsyncMock(return_value=raw_single_embedding),
        ) as mock_feature_extraction:
            embedding = hf_inference_api_embeddings.get_query_embedding("test")
        assert isinstance(embedding, list)
        assert len(embedding) == 1024
        assert np.all(
            np.array(embedding, dtype=raw_single_embedding.dtype)
            == raw_single_embedding[0, 0]
        )
        mock_feature_extraction.assert_awaited_once_with("test")

        hf_inference_api_embeddings.pooling = Pooling.MEAN
        with patch.object(
            hf_inference_api_embeddings._async_client,
            "feature_extraction",
            AsyncMock(return_value=raw_single_embedding),
        ) as mock_feature_extraction:
            embedding = hf_inference_api_embeddings.get_query_embedding("test")
        assert isinstance(embedding, list)
        assert len(embedding) == 1024
        assert np.all(
            np.array(embedding, dtype=raw_single_embedding.dtype)
            == raw_single_embedding[0].mean(axis=0)
        )
        mock_feature_extraction.assert_awaited_once_with("test")

    def test_serialization(
        self, hf_inference_api_embeddings: HuggingFaceInferenceAPIEmbeddings
    ) -> None:
        serialized = hf_inference_api_embeddings.to_dict()
        # Check Hugging Face Inference API base class specifics
        assert serialized["model_name"] == STUB_MODEL_NAME
        assert isinstance(serialized["context_window"], int)
        # Check Hugging Face Inference API Embeddings derived class specifics
        assert serialized["pooling"] == Pooling.CLS
