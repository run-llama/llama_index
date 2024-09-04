from llama_index.finetuning.mistralai import MistralAIFinetuneEngine
from llama_index.finetuning.types import BaseLLMFinetuneEngine


def test_class():
    names_of_base_classes = [b.__name__ for b in MistralAIFinetuneEngine.__mro__]
    assert BaseLLMFinetuneEngine.__name__ in names_of_base_classes
