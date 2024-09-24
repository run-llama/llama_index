from llama_index.packs.memary import MemaryChatAgentPack


def test_class():
    names_of_base_classes = [b.__name__ for b in MemaryChatAgentPack.__mro__]
    assert MemaryChatAgentPack.__name__ in names_of_base_classes
