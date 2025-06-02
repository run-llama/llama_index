from llama_index.agent.llm_compiler import (
    LLMCompilerAgentWorker,
)
from llama_index.core.agent.types import BaseAgentWorker


def test_classes():
    names_of_base_classes = [b.__name__ for b in LLMCompilerAgentWorker.__mro__]
    assert BaseAgentWorker.__name__ in names_of_base_classes
