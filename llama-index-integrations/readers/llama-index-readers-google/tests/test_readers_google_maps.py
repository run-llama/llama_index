import os
import unittest
from unittest.mock import patch, MagicMock

import pytest
from llama_index.readers.google import GoogleMapsTextSearchReader

mock_response_data = {
    "places": [
        {
            "formattedAddress": "123 Test St, Test City, TC 12345",
            "rating": 4.5,
            "displayName": "Test Place",
            "userRatingCount": 100,
            "reviews": [
                {
                    "text": {"text": "Great place!"},
                    "authorAttribution": {"displayName": "John Doe"},
                    "relativePublishTimeDescription": "2 days ago",
                    "rating": 5,
                }
            ],
        }
    ],
    "nextPageToken": "next_page_token_example",
}


class TestGoogleMapsTextSearchReader(unittest.TestCase):
    @patch.dict(os.environ, {"GOOGLE_MAPS_API_KEY": "test_api_key"})
    @patch("requests.post")
    def test_load_data(self, mock_post):
        # Mock API response
        mock_post.return_value.json.return_value = mock_response_data
        mock_post.return_value.status_code = 200

        reader = GoogleMapsTextSearchReader()
        documents = reader.load_data(text="Test", number_of_results=1)

        self.assertEqual(len(documents), 1)
        self.assertIn("Test Place", documents[0].text)
        self.assertIn("123 Test St, Test City, TC 12345", documents[0].text)
        self.assertIn("Great place!", documents[0].text)
        self.assertIn("John Doe", documents[0].text)
        self.assertIn("2 days ago", documents[0].text)

    @patch.dict(os.environ, {"GOOGLE_MAPS_API_KEY": "test_api_key"})
    def test_missing_api_key(self):
        # Unset API key
        os.environ.pop("GOOGLE_MAPS_API_KEY")

        with pytest.raises(ValueError):
            GoogleMapsTextSearchReader()

    @patch("requests.post")
    def test_load_data_with_pagination(self, mock_post):
        # Mock API response with pagination
        mock_post.side_effect = [
            MagicMock(status_code=200, json=MagicMock(return_value=mock_response_data)),
            MagicMock(status_code=200, json=MagicMock(return_value={"places": []})),
        ]

        reader = GoogleMapsTextSearchReader(api_key="test_api_key")
        documents = reader.load_data(text="Test", number_of_results=2)

        self.assertEqual(len(documents), 1)
        self.assertIn("Test Place", documents[0].text)

    @patch.dict(os.environ, {"GOOGLE_MAPS_API_KEY": "test_api_key"})
    @patch("requests.post")
    def test_lazy_load_data(self, mock_post):
        # Mock API response
        mock_post.return_value.json.return_value = mock_response_data
        mock_post.return_value.status_code = 200

        reader = GoogleMapsTextSearchReader()
        documents = list(reader.lazy_load_data(text="Test", number_of_results=1))

        self.assertEqual(len(documents), 1)
        self.assertIn("Test Place", documents[0].text)
