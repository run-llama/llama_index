from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.packs.tables import ChainOfTablePack, MixSelfConsistencyPack


def test_class():
    names_of_base_classes = [b.__name__ for b in ChainOfTablePack.__mro__]
    assert BaseLlamaPack.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in MixSelfConsistencyPack.__mro__]
    assert BaseLlamaPack.__name__ in names_of_base_classes
