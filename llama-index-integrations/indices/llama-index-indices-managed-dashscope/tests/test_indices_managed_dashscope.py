from llama_index.indices.managed.dashscope import DashScopeCloudIndex
from llama_index.core.indices.managed.base import BaseManagedIndex


def test_class():
    names_of_base_classes = [b.__name__ for b in DashScopeCloudIndex.__mro__]
    assert BaseManagedIndex.__name__ in names_of_base_classes
