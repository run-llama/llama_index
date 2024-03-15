from llama_index.core.readers.base import BaseReader
from llama_index.readers.gpt_repo import GPTRepoReader


def test_class():
    names_of_base_classes = [b.__name__ for b in GPTRepoReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
