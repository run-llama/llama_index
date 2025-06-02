from llama_index.llms.mistralai import MistralAI
from llama_index.multi_modal_llms.mistralai import MistralAIMultiModal


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in MistralAIMultiModal.__mro__]
    assert MistralAI.__name__ in names_of_base_classes


def test_init():
    m = MistralAIMultiModal(max_tokens=400, api_key="test")
    assert m.max_tokens == 400
