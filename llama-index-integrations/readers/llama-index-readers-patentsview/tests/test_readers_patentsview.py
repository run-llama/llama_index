"""Tests for Patentsview."""

import re
from unittest.mock import patch

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

import pytest
from llama_index.readers.patentsview import PatentsviewReader


@pytest.fixture
def api_key() -> str:
    """Test API key fixture."""
    return "test_api_key_123"


@pytest.fixture
def mock_json_response() -> dict:
    """Mock JSON response content."""
    return {
        "error": False,
        "count": 2,
        "total_hits": 2,
        "patents": [
            {
                "patent_id": "8848839",
                "patent_abstract": "Four score and seven years ago...",
            },
            {
                "patent_id": "10452978",
                "patent_abstract": "When in the course of human events...",
            },
        ],
    }


class TestPatentsviewReader:
    """Test cases for PatentsviewReader."""

    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        with pytest.raises(
            ValueError,
            match=re.escape("The API key [PATENTSVIEW_API_KEY] is required."),
        ):
            PatentsviewReader()

    def test_load_data_overload(self, api_key):
        """Test requesting more than 1000 patents"""
        loader = PatentsviewReader(api_key=api_key)
        too_many_patents = list(range(1001))
        with pytest.raises(ValueError, match="List patent number size is too large"):
            abstracts = loader.load_data(too_many_patents)

    @patch("llama_index.readers.patentsview.base.requests.post")
    def test_load_data(self, mock_post, api_key, mock_json_response):
        """Test reading patent abstracts"""
        mock_post.return_value.json.return_value = mock_json_response
        mock_post.return_value.status_code = 200

        loader = PatentsviewReader(api_key=api_key)
        patents = ["8848839", "10452978"]
        abstracts = loader.load_data(patents)
        assert len(abstracts) == 2
        assert isinstance(abstracts[0], Document)

    def test_class(self):
        names_of_base_classes = [b.__name__ for b in PatentsviewReader.__mro__]
        assert BaseReader.__name__ in names_of_base_classes
