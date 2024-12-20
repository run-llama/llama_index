from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.packs.gradio_agent_chat import GradioAgentChatPack


def test_class():
    names_of_base_classes = [b.__name__ for b in GradioAgentChatPack.__mro__]
    assert BaseLlamaPack.__name__ in names_of_base_classes
