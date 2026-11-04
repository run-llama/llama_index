"""Unit tests for LlamaIndex x402 CleanWeb Reader."""
import unittest
from unittest.mock import patch, MagicMock

from llama_index.readers.x402_cleanweb import X402CleanWebReader


class TestX402CleanWebReader(unittest.TestCase):
    def setUp(self):
        self.reader = X402CleanWebReader(
            gateway_url="https://test-gateway.x402.agent",
            vault_key="test-vault-secret"
        )

    def test_init_and_headers(self):
        self.assertEqual(self.reader.gateway_url, "https://test-gateway.x402.agent")
        self.assertEqual(self.reader.vault_key, "test-vault-secret")
        headers = self.reader._build_headers()
        self.assertEqual(headers["User-Agent"], "LlamaIndex-Community-x402/1.0")
        self.assertEqual(headers["X-Vault-Key"], "test-vault-secret")
        self.assertTrue(headers["X-Agent-Nonce"].startswith("llamaindex_"))

    @patch("requests.get")
    def test_load_data_success_single_url(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "url": "https://example.com/docs",
            "title": "Example High-Purity RAG Page",
            "markdown_content": "# High Purity RAG\nClean text content.",
            "security_audit": {
                "is_safe": True,
                "content_hash": "sha256_mock_hash_123"
            },
            "token_analytics": {
                "token_reduction_percent": 87.5
            },
            "spa_hydration": {
                "framework": "Next.js"
            },
            "word_count": 120,
            "estimated_reading_time_sec": 30
        }
        mock_get.return_value = mock_response

        docs = self.reader.load_data("https://example.com/docs")
        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertIn("Clean text content.", doc.text)
        self.assertEqual(doc.metadata["source"], "https://example.com/docs")
        self.assertEqual(doc.metadata["title"], "Example High-Purity RAG Page")
        self.assertEqual(doc.metadata["token_reduction_percent"], 87.5)
        self.assertEqual(doc.metadata["spa_framework"], "Next.js")
        self.assertEqual(doc.metadata["ingestion_engine"], "x402_cleanweb_studio")
        self.assertTrue(doc.metadata["is_safe"])

    @patch("requests.get")
    def test_load_data_402_payment_required(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 402
        mock_get.return_value = mock_response

        docs = self.reader.load_data(["https://premium.example.com"])
        self.assertEqual(len(docs), 1)
        self.assertIn("[x402 Payment Required]", docs[0].text)
        self.assertEqual(docs[0].metadata["error"], "402_payment_required")

    @patch("requests.get")
    def test_load_data_exception_handling(self, mock_get):
        mock_get.side_effect = TimeoutError("Connection timed out")

        docs = self.reader.load_data(["https://timeout.example.com"])
        self.assertEqual(len(docs), 1)
        self.assertIn("[CleanWeb Exception]", docs[0].text)
        self.assertIn("Connection timed out", docs[0].metadata["error"])


if __name__ == "__main__":
    unittest.main()

