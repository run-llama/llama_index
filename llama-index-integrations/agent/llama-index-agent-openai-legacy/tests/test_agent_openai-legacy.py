from llama_index.agent.openai_legacy import (
    BaseOpenAIAgent,
    ContextRetrieverOpenAIAgent,
    FnRetrieverOpenAIAgent,
    OpenAIAgent,
)
from llama_index.core.agent.types import BaseAgent


def test_classes():
    names_of_base_classes = [b.__name__ for b in OpenAIAgent.__mro__]
    assert BaseAgent.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in FnRetrieverOpenAIAgent.__mro__]
    assert BaseOpenAIAgent.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in ContextRetrieverOpenAIAgent.__mro__]
    assert BaseOpenAIAgent.__name__ in names_of_base_classes
