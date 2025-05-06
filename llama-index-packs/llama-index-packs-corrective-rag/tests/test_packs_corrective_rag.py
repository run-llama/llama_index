from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.packs.corrective_rag import CorrectiveRAGPack


def test_class():
    names_of_base_classes = [b.__name__ for b in CorrectiveRAGPack.__mro__]
    assert BaseLlamaPack.__name__ in names_of_base_classes
