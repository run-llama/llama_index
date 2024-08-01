from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.packs.cogniswitch_agent import CogniswitchAgentPack


def test_class():
    names_of_base_classes = [b.__name__ for b in CogniswitchAgentPack.__mro__]
    assert BaseLlamaPack.__name__ in names_of_base_classes
