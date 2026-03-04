import unittest
from llama_index.core.readers.base import BaseReader
from llama_index.readers.gitbook import SimpleGitbookReader
from typing import Dict, List, Optional


def test_class():
    """Tests if SimpleGitbookReader inherits from BaseReader."""
    names_of_base_classes = [b.__name__ for b in SimpleGitbookReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


class MockGitbookClient:
    """Mock class that simulates the GitBook API client."""

    def __init__(self, api_token: str, api_url: Optional[str] = None):
        self.pages_data = [
            {
                "id": "page1",
                "title": "Getting Started",
                "path": "/getting-started",
                "description": "Guide to get started",
                "parent": None,
            },
            {
                "id": "page2",
                "title": "Advanced Features",
                "path": "/advanced",
                "description": "Advanced usage guide",
                "parent": "page1",
            },
        ]

        self.page_content = {
            "page1": {
                "title": "Getting Started",
                "path": "/getting-started",
                "description": "Guide to get started",
                "parent": None,
                "markdown": "# Getting Started\n\nThis is a guide to get started.\n\n## Introduction\nLearn the basic usage.",
            },
            "page2": {
                "title": "Advanced Features",
                "path": "/advanced",
                "description": "Advanced usage guide",
                "parent": "page1",
                "markdown": "# Advanced Features\n\nThis document is for advanced users.\n\n## Detailed Settings\nLearn about advanced configuration options.",
            },
        }

    def list_pages(self, space_id: str) -> List[Dict]:
        """Returns list of pages in a given space."""
        if space_id == "non_existent_space":
            return []
        return self.pages_data

    def get_page(self, space_id: str, page_id: str) -> Dict:
        """Returns the content of a specific page."""
        return self.page_content.get(page_id, {"markdown": ""})

    def get_page_markdown(self, space_id, page_id) -> str:
        """Returns the content of a specific page in Markdown format."""
        page_content = self.get_page(space_id, page_id)
        return page_content.get("markdown")


class TestSimpleGitbookReader(unittest.TestCase):
    """Test cases for SimpleGitbookReader class."""

    def setUp(self):
        """Sets up test environment before each test case."""
        self.mock_client = MockGitbookClient("fake_token")
        self.reader = SimpleGitbookReader(api_token="fake_token")
        self.reader.client = self.mock_client

    def test_load_data_basic(self):
        """Tests basic data loading functionality."""
        docs = self.reader.load_data("space1")
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["path"], "/getting-started")
        self.assertEqual(docs[1].metadata["path"], "/advanced")

    def test_load_data_with_metadata(self):
        """Tests data loading with specified metadata fields."""
        docs = self.reader.load_data(
            "space1", metadata_names=["path", "title", "description", "parent"]
        )
        self.assertEqual(len(docs), 2)
        first_doc = docs[0]
        self.assertEqual(first_doc.metadata["title"], "Getting Started")
        self.assertEqual(first_doc.metadata["description"], "Guide to get started")
        self.assertIsNone(first_doc.metadata["parent"])

        second_doc = docs[1]
        self.assertEqual(second_doc.metadata["title"], "Advanced Features")
        self.assertEqual(second_doc.metadata["parent"], "page1")

    def test_load_data_invalid_space(self):
        """Tests behavior when loading data from non-existent space."""
        docs = self.reader.load_data("non_existent_space")
        self.assertEqual(len(docs), 0)

    def test_load_data_invalid_metadata(self):
        """Tests behavior when requesting invalid metadata field."""
        import pytest

        with pytest.raises(ValueError):
            self.reader.load_data("space1", metadata_names=["invalid_field"])

    def test_load_data_with_progress(self):
        """Tests data loading with progress bar enabled."""
        docs = self.reader.load_data("space1", show_progress=True)
        self.assertEqual(len(docs), 2)
