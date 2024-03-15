from llama_index.core.indices.managed.base import BaseManagedIndex
from llama_index.indices.managed.vectara import VectaraIndex


def test_class():
    names_of_base_classes = [b.__name__ for b in VectaraIndex.__mro__]
    assert BaseManagedIndex.__name__ in names_of_base_classes
