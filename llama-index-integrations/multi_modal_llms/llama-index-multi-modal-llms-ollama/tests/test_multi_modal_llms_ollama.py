from llama_index.core.llms import LLM
from llama_index.multi_modal_llms.ollama import OllamaMultiModal


def test_class():
    names_of_base_classes = [b.__name__ for b in OllamaMultiModal.__mro__]
    assert LLM.__name__ in names_of_base_classes
