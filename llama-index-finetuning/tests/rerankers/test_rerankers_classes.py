from llama_index.finetuning.types import BaseCohereRerankerFinetuningEngine
from llama_index.finetuning.rerankers import CohereRerankerFinetuneEngine


def test_classes():
    names_of_base_classes = [b.__name__ for b in CohereRerankerFinetuneEngine.__mro__]
    assert BaseCohereRerankerFinetuningEngine.__name__ in names_of_base_classes
