from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.sambanova import Sambaverse, SambaStudio


def test_embedding_class():
    # Check Sambaverse inherits from BaseLLM
    names_of_base_classes = [b.__name__ for b in Sambaverse.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes

    # Check SambaStudio inherits from BaseLLM
    names_of_base_classes = [b.__name__ for b in SambaStudio.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes
