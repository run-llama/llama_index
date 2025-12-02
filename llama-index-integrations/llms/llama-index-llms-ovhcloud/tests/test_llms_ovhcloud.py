from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.ovhcloud import OVHcloud


def test_text_inference_embedding_class():
    names_of_base_classes = [b.__name__ for b in OVHcloud.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes
