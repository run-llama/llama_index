from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.packs.rag_fusion_query_pipeline import RAGFusionPipelinePack


def test_class():
    names_of_base_classes = [b.__name__ for b in RAGFusionPipelinePack.__mro__]
    assert BaseLlamaPack.__name__ in names_of_base_classes
