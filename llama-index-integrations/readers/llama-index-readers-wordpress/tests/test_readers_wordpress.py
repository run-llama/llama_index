from llama_index.core.readers.base import BaseReader
from llama_index.readers.wordpress import WordpressReader


def test_class() -> None:
    names_of_base_classes = [b.__name__ for b in WordpressReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_allow_with_username_and_password() -> None:
    wordpress_reader = WordpressReader("http://example.com", "user", "pass")


def test_allow_without_username_and_password() -> None:
    wordpress_reader = WordpressReader("http://example.com")
