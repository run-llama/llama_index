from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.packs.fusion_retriever import (
    HybridFusionRetrieverPack,
    QueryRewritingRetrieverPack,
)


def test_class():
    names_of_base_classes = [b.__name__ for b in HybridFusionRetrieverPack.__mro__]
    assert BaseLlamaPack.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in QueryRewritingRetrieverPack.__mro__]
    assert BaseLlamaPack.__name__ in names_of_base_classes
