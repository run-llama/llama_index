from llama_index.finetuning.types import BaseLLMFinetuneEngine
from llama_index.finetuning.gradient import GradientFinetuneEngine


def test_class():
    names_of_base_classes = [b.__name__ for b in GradientFinetuneEngine.__mro__]
    assert BaseLLMFinetuneEngine.__name__ in names_of_base_classes
