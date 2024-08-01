from llama_index.finetuning.cross_encoders import CrossEncoderFinetuneEngine
from llama_index.finetuning.types import BaseCrossEncoderFinetuningEngine


def test_class():
    names_of_base_classes = [b.__name__ for b in CrossEncoderFinetuneEngine.__mro__]
    assert BaseCrossEncoderFinetuningEngine.__name__ in names_of_base_classes
