from llama_index.node_parser.alibabacloud_aisearch import AlibabaCloudAISearchNodeParser
from llama_index.core.node_parser.interface import NodeParser


def test_class():
    names_of_base_classes = [b.__name__ for b in AlibabaCloudAISearchNodeParser.__mro__]
    assert NodeParser.__name__ in names_of_base_classes
