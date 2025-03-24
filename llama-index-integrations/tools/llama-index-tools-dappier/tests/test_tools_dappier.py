from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.dappier import (
    DappierAIRecommendationsToolSpec,
    DappierRealTimeSearchToolSpec,
)


def test_class():
    names_of_base_classes = [
        b.__name__ for b in DappierAIRecommendationsToolSpec.__mro__
    ]
    assert BaseToolSpec.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in DappierRealTimeSearchToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes
