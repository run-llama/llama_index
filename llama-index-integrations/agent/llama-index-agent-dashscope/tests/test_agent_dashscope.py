from llama_index.core.agent.types import BaseAgent

from llama_index.agent.dashscope import (
    DashScopeAgent,
)


def test_classes():
    names_of_base_classes = [b.__name__ for b in DashScopeAgent.__mro__]
    assert BaseAgent.__name__ in names_of_base_classes
