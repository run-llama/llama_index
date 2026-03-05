import asyncio
from unittest import TestCase, mock
from unittest.mock import MagicMock, AsyncMock

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.postprocessor.google_rerank import GoogleRerank


def _create_reranker(**kwargs):
    """Create a GoogleRerank instance with mocked clients."""
    with mock.patch(
        "llama_index.postprocessor.google_rerank.base.discoveryengine",
        create=True,
    ):
        with mock.patch("google.auth.default", return_value=(None, "test-project")):
            with mock.patch(
                "google.cloud.discoveryengine_v1.RankServiceClient"
            ) as mock_client_cls, mock.patch(
                "google.cloud.discoveryengine_v1.RankServiceAsyncClient"
            ) as mock_async_client_cls:
                mock_client_cls.return_value = MagicMock()
                mock_async_client_cls.return_value = MagicMock()
                reranker = GoogleRerank(
                    project_id="test-project",
                    **kwargs,
                )
    return reranker


def _make_mock_response(results):
    """Create a mock rank response with records."""
    mock_records = []
    for r in results:
        record = MagicMock()
        record.id = str(r["index"])
        record.score = r["score"]
        record.content = r.get("content", "")
        mock_records.append(record)

    mock_response = MagicMock()
    mock_response.records = mock_records
    return mock_response


class TestGoogleRerank(TestCase):
    def test_class(self):
        names_of_base_classes = [b.__name__ for b in GoogleRerank.__mro__]
        self.assertIn(BaseNodePostprocessor.__name__, names_of_base_classes)

    def test_google_rerank(self):
        reranker = _create_reranker(top_n=2)

        mock_response = _make_mock_response([
            {"index": 2, "score": 0.95},
            {"index": 3, "score": 0.80},
        ])
        reranker._client.rank.return_value = mock_response

        input_nodes = [
            NodeWithScore(node=TextNode(id_="1", text="first 1")),
            NodeWithScore(node=TextNode(id_="2", text="first 2")),
            NodeWithScore(node=TextNode(id_="3", text="last 1")),
            NodeWithScore(node=TextNode(id_="4", text="last 2")),
        ]

        query_bundle = QueryBundle(query_str="last")
        actual_nodes = reranker.postprocess_nodes(
            input_nodes, query_bundle=query_bundle
        )

        self.assertEqual(len(actual_nodes), 2)
        self.assertEqual(actual_nodes[0].node.get_content(), "last 1")
        self.assertAlmostEqual(actual_nodes[0].score, 0.95)
        self.assertEqual(actual_nodes[1].node.get_content(), "last 2")
        self.assertAlmostEqual(actual_nodes[1].score, 0.80)

        reranker._client.rank.assert_called_once()

    def test_google_rerank_async(self):
        reranker = _create_reranker(top_n=2)

        mock_response = _make_mock_response([
            {"index": 1, "score": 0.90},
            {"index": 0, "score": 0.70},
        ])
        reranker._async_client.rank = AsyncMock(return_value=mock_response)

        input_nodes = [
            NodeWithScore(node=TextNode(id_="1", text="hello world")),
            NodeWithScore(node=TextNode(id_="2", text="goodbye world")),
        ]

        query_bundle = QueryBundle(query_str="goodbye")
        actual_nodes = asyncio.run(
            reranker.apostprocess_nodes(input_nodes, query_bundle=query_bundle)
        )

        self.assertEqual(len(actual_nodes), 2)
        self.assertEqual(actual_nodes[0].node.get_content(), "goodbye world")
        self.assertAlmostEqual(actual_nodes[0].score, 0.90)
        self.assertEqual(actual_nodes[1].node.get_content(), "hello world")
        self.assertAlmostEqual(actual_nodes[1].score, 0.70)

        reranker._async_client.rank.assert_called_once()

    def test_google_rerank_empty_nodes(self):
        reranker = _create_reranker()

        query_bundle = QueryBundle(query_str="test")
        actual_nodes = reranker.postprocess_nodes([], query_bundle=query_bundle)

        self.assertEqual(actual_nodes, [])
        reranker._client.rank.assert_not_called()

    def test_google_rerank_top_n_clamping(self):
        reranker = _create_reranker(top_n=10)

        mock_response = _make_mock_response([
            {"index": 0, "score": 0.9},
        ])
        reranker._client.rank.return_value = mock_response

        input_nodes = [
            NodeWithScore(node=TextNode(id_="1", text="only node")),
        ]

        query_bundle = QueryBundle(query_str="test")
        reranker.postprocess_nodes(input_nodes, query_bundle=query_bundle)

        call_args = reranker._client.rank.call_args
        request = call_args.kwargs.get("request") or call_args.args[0]
        self.assertEqual(request.top_n, 1)

    def test_google_rerank_no_query_raises(self):
        reranker = _create_reranker()

        input_nodes = [
            NodeWithScore(node=TextNode(id_="1", text="test")),
        ]

        with self.assertRaises(ValueError, msg="Missing query bundle"):
            reranker.postprocess_nodes(input_nodes)
