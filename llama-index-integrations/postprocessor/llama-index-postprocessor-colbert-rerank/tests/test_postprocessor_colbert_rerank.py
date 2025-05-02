from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.postprocessor.colbert_rerank import ColbertRerank


def test_class():
    names_of_base_classes = [b.__name__ for b in ColbertRerank.__mro__]
    assert BaseNodePostprocessor.__name__ in names_of_base_classes


def test_init():
    m = ColbertRerank(top_n=10)

    assert m.model == "colbert-ir/colbertv2.0"
    assert m.top_n == 10
