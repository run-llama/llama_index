from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.postprocessor.xinference_rerank import XinferenceRerank


def test_class():
    names_of_base_classes = [b.__name__ for b in XinferenceRerank.__mro__]
    assert BaseNodePostprocessor.__name__ in names_of_base_classes
