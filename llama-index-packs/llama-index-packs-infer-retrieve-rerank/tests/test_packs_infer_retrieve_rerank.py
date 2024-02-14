from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.packs.infer_retrieve_rerank import InferRetrieveRerankPack


def test_class():
    names_of_base_classes = [b.__name__ for b in InferRetrieveRerankPack.__mro__]
    assert BaseLlamaPack.__name__ in names_of_base_classes
