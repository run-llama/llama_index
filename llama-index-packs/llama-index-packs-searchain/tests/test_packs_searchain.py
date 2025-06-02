from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.packs.searchain import SearChainPack
from llama_index.packs.searchain.base import _normalize_answer, _match_or_not


def test_class():
    names_of_base_classes = [b.__name__ for b in SearChainPack.__mro__]
    assert BaseLlamaPack.__name__ in names_of_base_classes


def test_normalize_answer():
    assert _normalize_answer("  The Apple!! ") == "apple"


def test_match_or_not():
    assert _match_or_not("The Apple Pie", "apple")
    assert not _match_or_not("Banana", "apple")
