from llama_index.node_parser.dashscope import DashScopeJsonNodeParser
from llama_index.core.node_parser.relational.base_element import BaseElementNodeParser


def test_class():
    names_of_base_classes = [b.__name__ for b in DashScopeJsonNodeParser.__mro__]
    assert BaseElementNodeParser.__name__ in names_of_base_classes
