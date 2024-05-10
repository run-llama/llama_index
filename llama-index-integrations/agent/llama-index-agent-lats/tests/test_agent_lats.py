from llama_index.agent.lats import (
    LATSAgentWorker,
)
from llama_index.core.agent.types import BaseAgentWorker


def test_classes():
    names_of_base_classes = [b.__name__ for b in LATSAgentWorker.__mro__]
    assert BaseAgentWorker.__name__ in names_of_base_classes
