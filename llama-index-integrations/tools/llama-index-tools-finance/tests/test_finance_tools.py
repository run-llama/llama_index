from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.finance import FinanceAgentToolSpec


def test_class():
    name_of_base_classes = [b.__name__ for b in FinanceAgentToolSpec.__mro__]
    assert BaseToolSpec.__name__ in name_of_base_classes
