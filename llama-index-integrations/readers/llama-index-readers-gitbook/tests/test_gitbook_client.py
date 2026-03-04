import unittest
import json
import os
from unittest.mock import patch, Mock
from llama_index.readers.gitbook.gitbook_client import GitbookClient
import requests


class TestGitbookClient(unittest.TestCase):
    def setUp(self):
        """Sets up test environment before each test case."""
        self.client = GitbookClient("test_token")
        self.space_id = "test_space"
        self.page_id = "test_page"
        self.fixtures_path = os.path.join(os.path.dirname(__file__), "fixtures")

    def load_fixture(self, filename):
        """Helper method to load test data files."""
        with open(os.path.join(self.fixtures_path, filename), encoding="utf-8") as f:
            return json.load(f)

    @patch("requests.get")
    def test_get_space(self, mock_get):
        # Load test data
        mock_data = self.load_fixture("space_response.json")

        # Set up mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_data
        mock_get.return_value = mock_response

        response = self.client.get_space(self.space_id)

        # Verify
        mock_get.assert_called_once_with(
            f"{self.client.base_url}/spaces/{self.space_id}",
            headers=self.client.headers,
        )
        self.assertEqual(response["title"], mock_data["title"])
        self.assertEqual(response["id"], mock_data["id"])

    @patch("requests.get")
    def test_list_pages(self, mock_get):
        # Load test data
        space_data = self.load_fixture("space_response.json")
        pages_data = self.load_fixture("pages_response.json")

        # Set up space info response
        space_response = Mock()
        space_response.status_code = 200
        space_response.json.return_value = space_data

        # Set up pages list response
        pages_response = Mock()
        pages_response.status_code = 200
        pages_response.json.return_value = pages_data

        mock_get.side_effect = [space_response, pages_response]

        response = self.client.list_pages(self.space_id)

        # Verify
        self.assertEqual(len(response), 2)
        self.assertEqual(
            response[0]["title"],
            f"{space_data['title']} > {pages_data['pages'][0]['title']}",
        )
        self.assertEqual(response[0]["id"], pages_data["pages"][0]["id"])

    @patch("requests.get")
    def test_get_page_markdown(self, mock_get):
        # Load test data
        mock_data = self.load_fixture("page_content_response.json")

        # Set up mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_data
        mock_get.return_value = mock_response

        content = self.client.get_page_markdown(self.space_id, self.page_id)

        # Verify
        mock_get.assert_called_once_with(
            f"{self.client.base_url}/spaces/{self.space_id}/content/page/{self.page_id}?format=markdown",
            headers=self.client.headers,
        )
        self.assertEqual(content, mock_data["markdown"])

    @patch("requests.get")
    def test_error_handling(self, mock_get):
        """Tests error handling for API requests."""
        # Set up mock error response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.ok = False
        mock_response.reason = "Unauthorized"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_get.return_value = mock_response

        # Verify error handling
        import pytest

        with pytest.raises(Exception) as context:
            self.client.get_space(self.space_id)

        error_message = str(context.value)
        assert "Error" in error_message
        assert "401" in error_message
        assert "Unauthorized" in error_message


if __name__ == "__main__":
    unittest.main()
