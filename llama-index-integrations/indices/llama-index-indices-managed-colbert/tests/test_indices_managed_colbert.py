from llama_index.core.indices.base import BaseIndex
from llama_index.indices.managed.colbert import ColbertIndex


def test_class():
    names_of_base_classes = [b.__name__ for b in ColbertIndex.__mro__]
    assert BaseIndex.__name__ in names_of_base_classes
