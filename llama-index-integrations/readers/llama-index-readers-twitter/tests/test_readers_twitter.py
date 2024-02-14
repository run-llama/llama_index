from llama_index.core.readers.base import BasePydanticReader
from llama_index.readers.twitter import TwitterTweetReader


def test_class():
    names_of_base_classes = [b.__name__ for b in TwitterTweetReader.__mro__]
    assert BasePydanticReader.__name__ in names_of_base_classes
