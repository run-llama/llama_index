from llama_index.core.indices.base import BaseIndex
from llama_index.indices.managed.bge_m3 import BGEM3Index


def test_class():
    names_of_base_classes = [b.__name__ for b in BGEM3Index.__mro__]
    assert BaseIndex.__name__ in names_of_base_classes
