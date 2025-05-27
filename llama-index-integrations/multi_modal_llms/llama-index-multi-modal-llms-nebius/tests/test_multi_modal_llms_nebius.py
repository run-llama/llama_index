from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.multi_modal_llms.nebius import NebiusMultiModal


def test_multi_modal_class():
    names_of_base_classes = [b.__name__ for b in NebiusMultiModal.__mro__]
    assert OpenAIMultiModal.__name__ in names_of_base_classes
