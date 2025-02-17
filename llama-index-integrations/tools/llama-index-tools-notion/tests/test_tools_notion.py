import os

import pytest
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.notion import NotionToolSpec

# Get yourself a page id and database id from your notion account
# Refer to the page: https://developers.notion.com/docs/create-a-notion-integration#give-your-integration-page-permissions

page_ids = ["17d66c19670f80c5aaddfb8a0a449179"]  # replace with your page id
database_ids = ["16066c19-670f-801d-adb8-fa9d1cdaa053"]  # replace with your database id


def test_class():
    names_of_base_classes = [b.__name__ for b in NotionToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


@pytest.mark.skipif(
    "NOTION_INTEGRATION_TOKEN" not in os.environ,
    reason="NOTION_INTEGRATION_TOKEN is not set",
)
def test_load_data_with_page_ids():
    tool = NotionToolSpec()
    content = tool.load_data(page_ids=page_ids)
    assert content


@pytest.mark.skipif(
    "NOTION_INTEGRATION_TOKEN" not in os.environ,
    reason="NOTION_INTEGRATION_TOKEN is not set",
)
def test_load_data_with_database_ids():
    tool = NotionToolSpec()
    content = tool.load_data(database_ids=database_ids)
    assert content


@pytest.mark.skipif(
    "NOTION_INTEGRATION_TOKEN" not in os.environ,
    reason="NOTION_INTEGRATION_TOKEN is not set",
)
def test_load_data_with_page_ids_and_database_ids():
    tool = NotionToolSpec()
    content = tool.load_data(page_ids=page_ids, database_ids=database_ids)
    assert content


@pytest.mark.skipif(
    "NOTION_INTEGRATION_TOKEN" not in os.environ,
    reason="NOTION_INTEGRATION_TOKEN is not set",
)
def test_search_data():
    tool = NotionToolSpec()
    result = tool.search_data(query="Website")  # replace with your search query
    assert len(result) > 0
