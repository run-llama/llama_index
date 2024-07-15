from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.cleanlab import CleanlabTLM


def test_llms_cleanlab():
    names_of_base_classes = [b.__name__ for b in CleanlabTLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes
