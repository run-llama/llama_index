from llama_index.core.multi_modal_llms.base import MultiModalLLM
from llama_index.multi_modal_llms.dashscope import DashScopeMultiModal


def test_class():
    names_of_base_classes = [b.__name__ for b in DashScopeMultiModal.__mro__]
    assert MultiModalLLM.__name__ in names_of_base_classes


def test_init():
    m = DashScopeMultiModal(top_k=2)
    assert m.top_k == 2
