from llama_index.agent.openai import (
    OpenAIAgent,
    OpenAIAgentWorker,
    OpenAIAssistantAgent,
)
from llama_index.core.agent.types import BaseAgent, BaseAgentWorker


def test_classes():
    names_of_base_classes = [b.__name__ for b in OpenAIAgent.__mro__]
    assert BaseAgent.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in OpenAIAssistantAgent.__mro__]
    assert BaseAgent.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in OpenAIAgentWorker.__mro__]
    assert BaseAgentWorker.__name__ in names_of_base_classes
