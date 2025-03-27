import pytest
from unittest.mock import MagicMock, patch

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.dappier import DappierAIRecommendationsToolSpec


def test_class():
    names_of_base_classes = [
        b.__name__ for b in DappierAIRecommendationsToolSpec.__mro__
    ]
    assert BaseToolSpec.__name__ in names_of_base_classes


@pytest.fixture()
def dappier_client():
    return MagicMock()


@pytest.fixture()
def recommendations_tool(dappier_client):
    tool = DappierAIRecommendationsToolSpec(api_key="your-api-key")
    tool.client = dappier_client
    return tool


@pytest.fixture()
def success_response():
    result = MagicMock()
    result.title = "Test Title"
    result.author = "Test Author"
    result.pubdate = "2025-03-24"
    result.site = "Test Site"
    result.site_domain = "testsite.com"
    result.source_url = "http://testsite.com/article"
    result.image_url = "http://testsite.com/image.jpg"
    result.summary = "Test Summary"
    result.score = 0.99
    response = MagicMock()
    response.status = "success"
    response.response.results = [result]
    expected_output = (
        "Result 1:\n"
        "Title: Test Title\n"
        "Author: Test Author\n"
        "Published on: 2025-03-24\n"
        "Source: Test Site (testsite.com)\n"
        "URL: http://testsite.com/article\n"
        "Image URL: http://testsite.com/image.jpg\n"
        "Summary: Test Summary\n"
        "Score: 0.99\n\n"
    )
    return response, expected_output


@pytest.fixture()
def failure_response():
    response = MagicMock()
    response.status = "error"
    return response


class TestDappierAIRecommendationsToolSpec:
    def test_init_without_api_key_raises_value_error(self, monkeypatch, dappier_client):
        monkeypatch.delenv("DAPPIER_API_KEY", raising=False)
        with patch("dappier.Dappier", return_value=dappier_client):
            with pytest.raises(ValueError) as excinfo:
                DappierAIRecommendationsToolSpec()
            assert "API key is required" in str(excinfo.value)

    def test_get_sports_news_recommendations_success(
        self, recommendations_tool, dappier_client, success_response
    ):
        response, expected_output = success_response
        dappier_client.get_ai_recommendations.return_value = response
        result = recommendations_tool.get_sports_news_recommendations(
            "sports query",
            similarity_top_k=5,
            ref="example.com",
            num_articles_ref=2,
            search_algorithm="trending",
        )
        assert result == expected_output
        dappier_client.get_ai_recommendations.assert_called_once_with(
            query="sports query",
            data_model_id="dm_01j0pb465keqmatq9k83dthx34",
            similarity_top_k=5,
            ref="example.com",
            num_articles_ref=2,
            search_algorithm="trending",
        )

    def test_get_sports_news_recommendations_failure(
        self, recommendations_tool, dappier_client, failure_response
    ):
        dappier_client.get_ai_recommendations.return_value = failure_response
        result = recommendations_tool.get_sports_news_recommendations("sports query")
        assert result == "The API response was not successful."

    def test_get_lifestyle_news_recommendations_success(
        self, recommendations_tool, dappier_client, success_response
    ):
        response, expected_output = success_response
        dappier_client.get_ai_recommendations.return_value = response
        result = recommendations_tool.get_lifestyle_news_recommendations(
            "lifestyle query",
            similarity_top_k=8,
            ref="lifestyle.com",
            num_articles_ref=1,
            search_algorithm="semantic",
        )
        assert result == expected_output
        dappier_client.get_ai_recommendations.assert_called_once_with(
            query="lifestyle query",
            data_model_id="dm_01j0q82s4bfjmsqkhs3ywm3x6y",
            similarity_top_k=8,
            ref="lifestyle.com",
            num_articles_ref=1,
            search_algorithm="semantic",
        )

    def test_get_lifestyle_news_recommendations_failure(
        self, recommendations_tool, dappier_client, failure_response
    ):
        dappier_client.get_ai_recommendations.return_value = failure_response
        result = recommendations_tool.get_lifestyle_news_recommendations(
            "lifestyle query"
        )
        assert result == "The API response was not successful."

    def test_get_iheartdogs_recommendations_success(
        self, recommendations_tool, dappier_client, success_response
    ):
        response, expected_output = success_response
        dappier_client.get_ai_recommendations.return_value = response
        result = recommendations_tool.get_iheartdogs_recommendations(
            "dog query",
            similarity_top_k=3,
            ref="dogsite.com",
            num_articles_ref=0,
            search_algorithm="most_recent",
        )
        assert result == expected_output
        dappier_client.get_ai_recommendations.assert_called_once_with(
            query="dog query",
            data_model_id="dm_01j1sz8t3qe6v9g8ad102kvmqn",
            similarity_top_k=3,
            ref="dogsite.com",
            num_articles_ref=0,
            search_algorithm="most_recent",
        )

    def test_get_iheartdogs_recommendations_failure(
        self, recommendations_tool, dappier_client, failure_response
    ):
        dappier_client.get_ai_recommendations.return_value = failure_response
        result = recommendations_tool.get_iheartdogs_recommendations("dog query")
        assert result == "The API response was not successful."

    def test_get_iheartcats_recommendations_success(
        self, recommendations_tool, dappier_client, success_response
    ):
        response, expected_output = success_response
        dappier_client.get_ai_recommendations.return_value = response
        result = recommendations_tool.get_iheartcats_recommendations(
            "cat query",
            similarity_top_k=7,
            ref="catsite.com",
            num_articles_ref=1,
            search_algorithm="most_recent_semantic",
        )
        assert result == expected_output
        dappier_client.get_ai_recommendations.assert_called_once_with(
            query="cat query",
            data_model_id="dm_01j1sza0h7ekhaecys2p3y0vmj",
            similarity_top_k=7,
            ref="catsite.com",
            num_articles_ref=1,
            search_algorithm="most_recent_semantic",
        )

    def test_get_iheartcats_recommendations_failure(
        self, recommendations_tool, dappier_client, failure_response
    ):
        dappier_client.get_ai_recommendations.return_value = failure_response
        result = recommendations_tool.get_iheartcats_recommendations("cat query")
        assert result == "The API response was not successful."

    def test_get_greenmonster_recommendations_success(
        self, recommendations_tool, dappier_client, success_response
    ):
        response, expected_output = success_response
        dappier_client.get_ai_recommendations.return_value = response
        result = recommendations_tool.get_greenmonster_recommendations(
            "greenmonster query",
            similarity_top_k=4,
            ref="green.com",
            num_articles_ref=2,
            search_algorithm="trending",
        )
        assert result == expected_output
        dappier_client.get_ai_recommendations.assert_called_once_with(
            query="greenmonster query",
            data_model_id="dm_01j5xy9w5sf49bm6b1prm80m27",
            similarity_top_k=4,
            ref="green.com",
            num_articles_ref=2,
            search_algorithm="trending",
        )

    def test_get_greenmonster_recommendations_failure(
        self, recommendations_tool, dappier_client, failure_response
    ):
        dappier_client.get_ai_recommendations.return_value = failure_response
        result = recommendations_tool.get_greenmonster_recommendations(
            "greenmonster query"
        )
        assert result == "The API response was not successful."

    def test_get_wishtv_recommendations_success(
        self, recommendations_tool, dappier_client, success_response
    ):
        response, expected_output = success_response
        dappier_client.get_ai_recommendations.return_value = response
        result = recommendations_tool.get_wishtv_recommendations(
            "tv query",
            similarity_top_k=6,
            ref="tv.com",
            num_articles_ref=3,
            search_algorithm="semantic",
        )
        assert result == expected_output
        dappier_client.get_ai_recommendations.assert_called_once_with(
            query="tv query",
            data_model_id="dm_01jagy9nqaeer9hxx8z1sk1jx6",
            similarity_top_k=6,
            ref="tv.com",
            num_articles_ref=3,
            search_algorithm="semantic",
        )

    def test_get_wishtv_recommendations_failure(
        self, recommendations_tool, dappier_client, failure_response
    ):
        dappier_client.get_ai_recommendations.return_value = failure_response
        result = recommendations_tool.get_wishtv_recommendations("tv query")
        assert result == "The API response was not successful."

    def test_get_nine_and_ten_news_recommendations_success(
        self, recommendations_tool, dappier_client, success_response
    ):
        response, expected_output = success_response
        dappier_client.get_ai_recommendations.return_value = response
        result = recommendations_tool.get_nine_and_ten_news_recommendations(
            "local news query",
            similarity_top_k=9,
            ref="localnews.com",
            num_articles_ref=5,
            search_algorithm="most_recent_semantic",
        )
        assert result == expected_output
        dappier_client.get_ai_recommendations.assert_called_once_with(
            query="local news query",
            data_model_id="dm_01jhtt138wf1b9j8jwswye99y5",
            similarity_top_k=9,
            ref="localnews.com",
            num_articles_ref=5,
            search_algorithm="most_recent_semantic",
        )

    def test_get_nine_and_ten_news_recommendations_failure(
        self, recommendations_tool, dappier_client, failure_response
    ):
        dappier_client.get_ai_recommendations.return_value = failure_response
        result = recommendations_tool.get_nine_and_ten_news_recommendations(
            "local news query"
        )
        assert result == "The API response was not successful."
