from llama_index.core.readers.base import BaseReader
from llama_index.readers.feishu_docs import FeishuDocsReader


def test_class():
    names_of_base_classes = [b.__name__ for b in FeishuDocsReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
