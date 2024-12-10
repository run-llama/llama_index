from llama_index.core.readers.base import BaseReader
from llama_index.readers.snscrape_twitter import SnscrapeTwitterReader


def test_class():
    names_of_base_classes = [b.__name__ for b in SnscrapeTwitterReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
