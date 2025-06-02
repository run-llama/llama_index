import pytest
from typing import List
from unittest.mock import patch, MagicMock

from llama_index.embeddings.ibm import WatsonxEmbeddings


class TestWasonxLLMInference:
    TEST_URL = "https://us-south.ml.cloud.ibm.com"
    TEST_APIKEY = "apikey"
    TEST_PROJECT_ID = "project_id"

    TEST_MODEL = "test_model"

    def mock_embed_query(self) -> List[float]:
        return [-0.053358648, -0.009175377, -0.025022397]

    def mock_embed_texts(self) -> List[List[float]]:
        return [
            [-0.053358648, -0.009175377, -0.025022397],
            [-0.053358648, -0.009175377, -0.025022397],
        ]

    def test_initialization(self) -> None:
        with pytest.raises(ValueError, match=r"^Did not find") as e_info:
            _ = WatsonxEmbeddings(
                model_id=self.TEST_MODEL, project_id=self.TEST_PROJECT_ID
            )

        # Cloud scenario
        with pytest.raises(
            ValueError, match=r"^Did not find 'apikey' or 'token',"
        ) as e_info:
            _ = WatsonxEmbeddings(
                model_id=self.TEST_MODEL,
                url=self.TEST_URL,
                project_id=self.TEST_PROJECT_ID,
            )

        # CPD scenario
        with pytest.raises(ValueError, match=r"^Did not find instance_id") as e_info:
            _ = WatsonxEmbeddings(
                model_id=self.TEST_MODEL,
                token="123",
                url="cpd-instance",
                project_id=self.TEST_PROJECT_ID,
            )

    @patch("llama_index.embeddings.ibm.base.Embeddings")
    def test_get_query_embedding(self, MockEmbedding: MagicMock) -> None:
        mock_instance = MockEmbedding.return_value
        mock_instance.embed_query.return_value = self.mock_embed_query()

        embed = WatsonxEmbeddings(
            model_id=self.TEST_MODEL,
            url=self.TEST_URL,
            apikey=self.TEST_APIKEY,
            project_id=self.TEST_PROJECT_ID,
        )

        assert embed.get_query_embedding(query="TEST") == self.mock_embed_query()

    @patch("llama_index.embeddings.ibm.base.Embeddings")
    def test_get_texts_embedding(self, MockEmbedding: MagicMock) -> None:
        mock_instance = MockEmbedding.return_value
        mock_instance.embed_documents.return_value = self.mock_embed_texts()

        embed = WatsonxEmbeddings(
            model_id=self.TEST_MODEL,
            url=self.TEST_URL,
            apikey=self.TEST_APIKEY,
            project_id=self.TEST_PROJECT_ID,
        )

        assert (
            embed.get_text_embedding_batch(texts=["TEST1", "TEST2"])
            == self.mock_embed_texts()
        )

    @pytest.mark.asyncio
    @patch("llama_index.embeddings.ibm.base.Embeddings")
    async def test_get_query_embedding_async(self, MockEmbedding: MagicMock) -> None:
        mock_instance = MockEmbedding.return_value
        mock_instance.embed_query.return_value = self.mock_embed_query()

        embed = WatsonxEmbeddings(
            model_id=self.TEST_MODEL,
            url=self.TEST_URL,
            apikey=self.TEST_APIKEY,
            project_id=self.TEST_PROJECT_ID,
        )
        response = await embed.aget_text_embedding("TEST1")
        assert response == self.mock_embed_query()
