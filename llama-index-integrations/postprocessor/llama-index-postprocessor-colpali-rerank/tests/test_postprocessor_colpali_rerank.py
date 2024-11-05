from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.postprocessor.colpali_rerank import ColPaliRerank


def test_class():
    names_of_base_classes = [b.__name__ for b in ColPlaiRerank.__mro__]
    assert BaseNodePostprocessor.__name__ in names_of_base_classes


def test_init():
    m = ColPaliRerank(top_n=10)

    assert m.model == "vidore/colpali"
    assert m.top_n == 10
