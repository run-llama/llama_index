from unittest.mock import MagicMock, patch

from llama_index.core.readers.base import BaseReader
from llama_index.readers.stackoverflow import StackoverflowReader


def test_class():
    names_of_base_classes = [b.__name__ for b in StackoverflowReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def _page(has_more: bool) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {
        "has_more": has_more,
        "items": [
            {
                "title": "How do I test this?",
                "link": "https://example.com/q/1",
                "score": 1,
                "last_activity_date": 1700000000,
                "creation_date": 1700000000,
                "post_id": 1,
                "body_markdown": "Question body",
                "owner": {"display_name": "someone", "user_id": 7},
            }
        ],
    }
    response.content = b'{"has_more": false, "items": []}'
    return response


@patch("llama_index.readers.stackoverflow.base.rate_limited_get")
def test_load_data_without_cache_dir(mock_get):
    mock_get.side_effect = [_page(True), _page(False)]

    reader = StackoverflowReader(api_key="key", team_name="team")
    documents = reader.load_data()

    assert len(documents) == 2
    assert documents[0].doc_id == "1"
    assert mock_get.call_count == 2
    assert "page=2" in mock_get.call_args_list[1].args[0]


@patch("llama_index.readers.stackoverflow.base.rate_limited_get")
def test_load_data_with_cache_dir_writes_pages(mock_get, tmp_path):
    mock_get.side_effect = [_page(False)]

    reader = StackoverflowReader(
        api_key="key", team_name="team", cache_dir=str(tmp_path)
    )
    documents = reader.load_data()

    assert len(documents) == 1
    assert (tmp_path / "posts_1.json").exists()


@patch("llama_index.readers.stackoverflow.base.rate_limited_get")
def test_load_data_articles_use_article_id(mock_get):
    response = MagicMock()
    response.json.return_value = {
        "has_more": False,
        "items": [
            {
                "title": "An article",
                "link": "https://example.com/a/5",
                "score": 0,
                "last_activity_date": 1700000000,
                "creation_date": 1700000000,
                "article_id": 5,
                "article_type": "knowledge-article",
                "body_markdown": "Article body",
            }
        ],
    }
    mock_get.side_effect = [response]

    reader = StackoverflowReader(api_key="key", team_name="team")
    documents = reader.load_data(doc_type="articles")

    assert len(documents) == 1
    assert documents[0].doc_id == "5"
