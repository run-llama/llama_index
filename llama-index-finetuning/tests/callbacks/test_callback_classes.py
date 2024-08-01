from llama_index.core.callbacks.base import BaseCallbackHandler
from llama_index.finetuning.callbacks.finetuning_handler import OpenAIFineTuningHandler


def test_class():
    names_of_base_classes = [b.__name__ for b in OpenAIFineTuningHandler.__mro__]
    assert BaseCallbackHandler.__name__ in names_of_base_classes
