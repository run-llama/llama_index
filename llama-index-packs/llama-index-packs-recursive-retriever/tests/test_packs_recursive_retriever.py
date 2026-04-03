from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.packs.recursive_retriever import (
    EmbeddedTablesUnstructuredRetrieverPack,
    RecursiveRetrieverSmallToBigPack,
)


def test_class():
    names_of_base_classes = [
        b.__name__ for b in EmbeddedTablesUnstructuredRetrieverPack.__mro__
    ]
    assert BaseLlamaPack.__name__ in names_of_base_classes

    names_of_base_classes = [
        b.__name__ for b in RecursiveRetrieverSmallToBigPack.__mro__
    ]
    assert BaseLlamaPack.__name__ in names_of_base_classes
