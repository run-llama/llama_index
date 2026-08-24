import json
import unittest
from unittest.mock import MagicMock, patch

from llama_index.tools.annolux import AnnoluxToolSpec


class TestLlamaIndexAnnolux(unittest.TestCase):

    def setUp(self):
        self.mock_response = {
            "query": "Rust tokio",
            "results": [
                {
                    "title": "Tokio Async Runtime",
                    "url": "https://tokio.rs",
                    "snippet": "Tokio is an asynchronous runtime for Rust.",
                    "fetched_at": "2026-08-20T10:00:00Z",
                    "domain": "tokio.rs",
                    "score": 0.98,
                }
            ],
            "total": 1,
        }

    @patch("requests.post")
    def test_tool_spec_search(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self.mock_response
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        spec = AnnoluxToolSpec(api_key="ann_test_123")
        docs = spec.annolux_search(query="Rust tokio", limit=1)

        self.assertEqual(len(docs), 1)
        self.assertIn("Tokio Async Runtime", docs[0].text)
        self.assertEqual(docs[0].extra_info["fetched_at"], "2026-08-20T10:00:00Z")
        self.assertEqual(docs[0].extra_info["domain"], "tokio.rs")


if __name__ == "__main__":
    unittest.main()
