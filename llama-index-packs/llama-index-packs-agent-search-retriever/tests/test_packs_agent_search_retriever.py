from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.packs.agent_search_retriever import AgentSearchRetrieverPack


def test_class():
    names_of_base_classes = [b.__name__ for b in AgentSearchRetrieverPack.__mro__]
    assert BaseLlamaPack.__name__ in names_of_base_classes
