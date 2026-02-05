"""Tests for the GoogleRerank postprocessor."""

from unittest import TestCase
from unittest.mock import MagicMock, patch

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.postprocessor.google_rerank import GoogleRerank
from llama_index.postprocessor.google_rerank.base import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
)


class TestGoogleRerank(TestCase):
    """Test cases for GoogleRerank postprocessor."""

    def test_class(self):
        """Test that GoogleRerank inherits from BaseNodePostprocessor."""
        names_of_base_classes = [b.__name__ for b in GoogleRerank.__mro__]
        self.assertIn(BaseNodePostprocessor.__name__, names_of_base_classes)

    def test_class_name(self):
        """Test that class_name returns the correct value."""
        self.assertEqual(GoogleRerank.class_name(), "GoogleRerank")

    def test_default_model_is_supported(self):
        """Test that the default model is in the supported models list."""
        self.assertIn(DEFAULT_MODEL, SUPPORTED_MODELS)

    def test_all_supported_models_have_properties(self):
        """Test that all supported models have max_tokens and languages properties."""
        for model_name, props in SUPPORTED_MODELS.items():
            self.assertIn("max_tokens", props, f"Model {model_name} missing max_tokens")
            self.assertIn("languages", props, f"Model {model_name} missing languages")

    @patch("llama_index.postprocessor.google_rerank.base.discoveryengine")
    def test_initialization_with_project_id(self, mock_discoveryengine):
        """Test that GoogleRerank initializes correctly with project_id."""
        mock_client = MagicMock()
        mock_client.ranking_config_path.return_value = "projects/test-project/locations/global/rankingConfigs/default_ranking_config"
        mock_discoveryengine.RankServiceClient.return_value = mock_client

        reranker = GoogleRerank(project_id="test-project", top_n=5)

        self.assertEqual(reranker.project_id, "test-project")
        self.assertEqual(reranker.top_n, 5)
        self.assertEqual(reranker.model, DEFAULT_MODEL)
        self.assertEqual(reranker.location, "global")

    @patch("llama_index.postprocessor.google_rerank.base.discoveryengine")
    def test_initialization_with_env_variable(self, mock_discoveryengine):
        """Test that GoogleRerank initializes with GOOGLE_CLOUD_PROJECT env var."""
        mock_client = MagicMock()
        mock_client.ranking_config_path.return_value = "projects/env-project/locations/global/rankingConfigs/default_ranking_config"
        mock_discoveryengine.RankServiceClient.return_value = mock_client

        with patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "env-project"}):
            reranker = GoogleRerank(top_n=3)

        self.assertEqual(reranker.project_id, "env-project")
        self.assertEqual(reranker.top_n, 3)

    def test_initialization_without_project_id_raises_error(self):
        """Test that initialization without project_id raises ValueError."""
        with patch.dict("os.environ", {}, clear=True):
            # Ensure GOOGLE_CLOUD_PROJECT is not set
            import os

            if "GOOGLE_CLOUD_PROJECT" in os.environ:
                del os.environ["GOOGLE_CLOUD_PROJECT"]

            with self.assertRaises(ValueError) as context:
                GoogleRerank()

            self.assertIn("project_id must be provided", str(context.exception))

    @patch("llama_index.postprocessor.google_rerank.base.discoveryengine")
    def test_rerank_postprocess_nodes(self, mock_discoveryengine):
        """Test the _postprocess_nodes method with mocked API."""
        # Setup mock client
        mock_client = MagicMock()
        mock_client.ranking_config_path.return_value = "projects/test-project/locations/global/rankingConfigs/default_ranking_config"
        mock_discoveryengine.RankServiceClient.return_value = mock_client

        # Setup mock response
        mock_record_1 = MagicMock()
        mock_record_1.id = "2"
        mock_record_1.score = 0.95

        mock_record_2 = MagicMock()
        mock_record_2.id = "0"
        mock_record_2.score = 0.85

        mock_response = MagicMock()
        mock_response.records = [mock_record_1, mock_record_2]
        mock_client.rank.return_value = mock_response

        reranker = GoogleRerank(project_id="test-project", top_n=2)
        reranker._client = mock_client

        # Create input nodes
        input_nodes = [
            NodeWithScore(node=TextNode(id_="1", text="Document about cats"), score=0.5),
            NodeWithScore(
                node=TextNode(id_="2", text="Document about dogs"), score=0.4
            ),
            NodeWithScore(
                node=TextNode(id_="3", text="Document about birds"), score=0.3
            ),
        ]

        query_bundle = QueryBundle(query_str="What animals make good pets?")

        # Call the method
        result = reranker._postprocess_nodes(input_nodes, query_bundle)

        # Verify results
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].score, 0.95)
        self.assertEqual(result[1].score, 0.85)
        # Check that the correct nodes are returned based on the mock response
        self.assertEqual(result[0].node.text, "Document about birds")  # index 2
        self.assertEqual(result[1].node.text, "Document about cats")  # index 0

        # Verify the API was called
        mock_client.rank.assert_called_once()

    @patch("llama_index.postprocessor.google_rerank.base.discoveryengine")
    def test_rerank_with_empty_nodes(self, mock_discoveryengine):
        """Test that _postprocess_nodes returns empty list for empty input."""
        mock_client = MagicMock()
        mock_client.ranking_config_path.return_value = "projects/test-project/locations/global/rankingConfigs/default_ranking_config"
        mock_discoveryengine.RankServiceClient.return_value = mock_client

        reranker = GoogleRerank(project_id="test-project", top_n=5)

        query_bundle = QueryBundle(query_str="test query")
        result = reranker._postprocess_nodes([], query_bundle)

        self.assertEqual(result, [])

    @patch("llama_index.postprocessor.google_rerank.base.discoveryengine")
    def test_rerank_without_query_bundle_raises_error(self, mock_discoveryengine):
        """Test that _postprocess_nodes raises error without query_bundle."""
        mock_client = MagicMock()
        mock_client.ranking_config_path.return_value = "projects/test-project/locations/global/rankingConfigs/default_ranking_config"
        mock_discoveryengine.RankServiceClient.return_value = mock_client

        reranker = GoogleRerank(project_id="test-project", top_n=5)

        input_nodes = [
            NodeWithScore(node=TextNode(id_="1", text="test"), score=0.5),
        ]

        with self.assertRaises(ValueError) as context:
            reranker._postprocess_nodes(input_nodes, query_bundle=None)

        self.assertIn("Missing query bundle", str(context.exception))

    @patch("llama_index.postprocessor.google_rerank.base.discoveryengine")
    def test_custom_model_selection(self, mock_discoveryengine):
        """Test that custom model can be specified."""
        mock_client = MagicMock()
        mock_client.ranking_config_path.return_value = "projects/test-project/locations/global/rankingConfigs/default_ranking_config"
        mock_discoveryengine.RankServiceClient.return_value = mock_client

        reranker = GoogleRerank(
            project_id="test-project",
            model="semantic-ranker-fast-004",
        )

        self.assertEqual(reranker.model, "semantic-ranker-fast-004")

    @patch("llama_index.postprocessor.google_rerank.base.discoveryengine")
    def test_custom_location(self, mock_discoveryengine):
        """Test that custom location can be specified."""
        mock_client = MagicMock()
        mock_client.ranking_config_path.return_value = "projects/test-project/locations/us-central1/rankingConfigs/default_ranking_config"
        mock_discoveryengine.RankServiceClient.return_value = mock_client

        reranker = GoogleRerank(
            project_id="test-project",
            location="us-central1",
        )

        self.assertEqual(reranker.location, "us-central1")
        # Verify client was created with the correct endpoint
        mock_discoveryengine.RankServiceClient.assert_called_once()
        call_kwargs = mock_discoveryengine.RankServiceClient.call_args
        self.assertIn("client_options", call_kwargs.kwargs)
