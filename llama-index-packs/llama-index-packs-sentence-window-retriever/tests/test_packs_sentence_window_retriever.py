from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.packs.sentence_window_retriever import SentenceWindowRetrieverPack


def test_class():
    names_of_base_classes = [b.__name__ for b in SentenceWindowRetrieverPack.__mro__]
    assert BaseLlamaPack.__name__ in names_of_base_classes
