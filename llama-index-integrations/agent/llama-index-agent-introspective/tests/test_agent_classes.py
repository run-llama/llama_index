from llama_index.agent.introspective import (
    IntrospectiveAgentWorker,
    SelfReflectionAgentWorker,
    ToolInteractiveReflectionAgentWorker,
)
from llama_index.core.agent.types import BaseAgentWorker


def test_classes():
    names_of_base_classes = [b.__name__ for b in IntrospectiveAgentWorker.__mro__]
    assert BaseAgentWorker.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in SelfReflectionAgentWorker.__mro__]
    assert BaseAgentWorker.__name__ in names_of_base_classes

    names_of_base_classes = [
        b.__name__ for b in ToolInteractiveReflectionAgentWorker.__mro__
    ]
    assert BaseAgentWorker.__name__ in names_of_base_classes
