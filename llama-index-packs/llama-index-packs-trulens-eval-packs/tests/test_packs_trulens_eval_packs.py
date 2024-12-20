from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.packs.trulens_eval_packs import (
    TruLensHarmlessPack,
    TruLensHelpfulPack,
    TruLensRAGTriadPack,
)


def test_class():
    names_of_base_classes = [b.__name__ for b in TruLensHarmlessPack.__mro__]
    assert BaseLlamaPack.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in TruLensHelpfulPack.__mro__]
    assert BaseLlamaPack.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in TruLensRAGTriadPack.__mro__]
    assert BaseLlamaPack.__name__ in names_of_base_classes
