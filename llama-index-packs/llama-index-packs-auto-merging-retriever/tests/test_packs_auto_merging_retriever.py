from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.packs.auto_merging_retriever import AutoMergingRetrieverPack


def test_class():
    names_of_base_classes = [b.__name__ for b in AutoMergingRetrieverPack.__mro__]
    assert BaseLlamaPack.__name__ in names_of_base_classes
