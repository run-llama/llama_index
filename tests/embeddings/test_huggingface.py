from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from llama_index.embeddings.utils import Pooling

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
        raw_embedding = np.random.rand(1, 1, 3, 1024)

        hf_inference_api_embeddings.pooling = Pooling.CLS
        with patch.object(
            hf_inference_api_embeddings._sync_client,
            "feature_extraction",
            return_value=raw_embedding,
        ):
            embedding = hf_inference_api_embeddings.embed_query(text="test")
        assert isinstance(embedding, list)
        assert len(embedding) == 1024
        assert np.all(
            np.array(embedding, dtype=raw_embedding.dtype) == raw_embedding[0, 0, 0]
        )

        hf_inference_api_embeddings.pooling = Pooling.MEAN
        with patch.object(
            hf_inference_api_embeddings._sync_client,
            "feature_extraction",
            return_value=raw_embedding,
        ):
            embedding = hf_inference_api_embeddings.embed_query(text="test")
        assert isinstance(embedding, list)
        assert len(embedding) == 1024
        assert np.all(
            np.array(embedding, dtype=raw_embedding.dtype)
            == raw_embedding[0, 0].mean(axis=0)
        )
