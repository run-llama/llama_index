from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.response_synthesizers.google import GoogleTextSynthesizer


def test_class():
    names_of_base_classes = [b.__name__ for b in GoogleTextSynthesizer.__mro__]
    assert BaseSynthesizer.__name__ in names_of_base_classes
