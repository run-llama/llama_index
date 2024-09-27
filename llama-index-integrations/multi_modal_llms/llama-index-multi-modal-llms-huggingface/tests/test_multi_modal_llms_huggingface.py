from llama_index.core.multi_modal_llms.base import MultiModalLLM
from llama_index.multi_modal_llms.huggingface import HuggingFaceMultiModal


def test_class():
    names_of_base_classes = [b.__name__ for b in HuggingFaceMultiModal.__mro__]
    assert MultiModalLLM.__name__ in names_of_base_classes
