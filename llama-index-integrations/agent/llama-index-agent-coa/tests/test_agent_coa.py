from llama_index.agent.coa import (
    CoAAgentWorker,
)
from llama_index.core.agent.types import BaseAgentWorker


def test_classes():
    names_of_base_classes = [b.__name__ for b in CoAAgentWorker.__mro__]
    assert BaseAgentWorker.__name__ in names_of_base_classes
