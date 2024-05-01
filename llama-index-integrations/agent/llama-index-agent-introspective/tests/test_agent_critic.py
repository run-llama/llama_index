from llama_index.agent.critic import (
    CriticAgentWorker,
)
from llama_index.core.agent.types import BaseAgentWorker


def test_classes():
    names_of_base_classes = [b.__name__ for b in CriticAgentWorker.__mro__]
    assert BaseAgentWorker.__name__ in names_of_base_classes
