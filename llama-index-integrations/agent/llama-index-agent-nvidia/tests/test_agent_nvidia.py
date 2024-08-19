from llama_index.agent.nvidia import NVIDIAAgent
from llama_index.core.agent.types import BaseAgent


def test_classes():
    names_of_base_classes = [b.__name__ for b in NVIDIAAgent.__mro__]
    assert BaseAgent.__name__ in names_of_base_classes
