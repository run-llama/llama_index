import pytest
import responses

from llama_index.core.readers.base import BaseReader
from llama_index.readers.wordpress import WordpressReader


@pytest.fixture()
def mocked_responses():
    with responses.RequestsMock() as rsps:
        yield rsps


def test_class() -> None:
    names_of_base_classes = [b.__name__ for b in WordpressReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_allow_with_username_and_password() -> None:
    wordpress_reader = WordpressReader("http://example.com", "user", "pass")


def test_allow_without_username_and_password() -> None:
    wordpress_reader = WordpressReader("http://example.com")


def test_retreive_pages_and_posts(mocked_responses) -> None:
    mocked_responses.get(
        "http://test.wordpress.org/wp-json/wp/v2/posts",
        content_type="application/json",
        body="""
                [{"id": 1,
                    "title": { "rendered": "foo" },
                    "link": "http://test.wordpress.org/posts/1",
                    "modified": "Never",
                    "content": { "rendered": "Lorem ipsum" }
                }]
            """,
    )
    mocked_responses.get(
        "http://test.wordpress.org/wp-json/wp/v2/pages",
        content_type="application/json",
        body="""
                [{"id": 1,
                    "title": { "rendered": "foo" },
                    "link": "http://test.wordpress.org/pages/1",
                    "modified": "Never",
                    "content": { "rendered": "Lorem ipsum" }
                }]
            """,
    )
    wordpress_reader = WordpressReader("http://test.wordpress.org")
    documents = wordpress_reader.load_data()


def test_retreive_pages(mocked_responses) -> None:
    mocked_responses.get(
        "http://test.wordpress.org/wp-json/wp/v2/pages",
        content_type="application/json",
        body="""
                [{"id": 1,
                    "title": { "rendered": "foo" },
                    "link": "http://test.wordpress.org/pages/1",
                    "modified": "Never",
                    "content": { "rendered": "Lorem ipsum" }
                }]
            """,
    )
    wordpress_reader = WordpressReader("http://test.wordpress.org", get_posts=False)
    documents = wordpress_reader.load_data()


def test_retreive_posts(mocked_responses) -> None:
    mocked_responses.get(
        "http://test.wordpress.org/wp-json/wp/v2/posts",
        content_type="application/json",
        body="""
                [{"id": 1,
                    "title": { "rendered": "foo" },
                    "link": "http://test.wordpress.org/posts/1",
                    "modified": "Never",
                    "content": { "rendered": "Lorem ipsum" }
                }]
            """,
    )
    wordpress_reader = WordpressReader("http://test.wordpress.org", get_pages=False)
    documents = wordpress_reader.load_data()
