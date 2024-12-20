from llama_index.core.output_parsers.base import ChainableOutputParser
from llama_index.output_parsers.langchain import LangchainOutputParser


def test_class():
    names_of_base_classes = [b.__name__ for b in LangchainOutputParser.__mro__]
    assert ChainableOutputParser.__name__ in names_of_base_classes
