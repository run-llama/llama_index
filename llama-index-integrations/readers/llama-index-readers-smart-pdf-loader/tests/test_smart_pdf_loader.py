from llama_index.readers.smart_pdf_loader import SmartPDFLoader
import unittest
import pkg_resources
from unittest.mock import patch, MagicMock


class TestLayoutReader(unittest.TestCase):
    @patch("llmsherpa.readers.file_reader.urllib3.PoolManager")
    def test_loader(self, mock_pool_manager):
        # Setup the mock behavior
        mock_response = MagicMock()
        with pkg_resources.resource_stream(
            __name__, "chunk_test_mock_response.json"
        ) as data_stream:
            mock_response.data = data_stream.read()
            mock_response.status = 200
            mock_pool_manager.return_value.request.return_value = mock_response
        # mock api url
        llmsherpa_api_url = "https://mockapiurl.com/api/document/developer/parseDocument?renderFormat=all"
        # mock pdf link
        pdf_url = "https://example.com/pdf/example.pdf"  # also allowed is a file path e.g. /home/downloads/xyz.pdf
        pdf_loader = SmartPDFLoader(llmsherpa_api_url=llmsherpa_api_url)
        documents = pdf_loader.load_data(pdf_url)
        self.assertEqual(len(documents), 5)
        self.assertEqual(documents[0].extra_info["chunk_type"], "list_item")


if __name__ == "__main__":
    unittest.main()
