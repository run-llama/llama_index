from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.core.utils import infer_torch_device


def test_class():
    names_of_base_classes = [b.__name__ for b in SentenceTransformerRerank.__mro__]
    assert BaseNodePostprocessor.__name__ in names_of_base_classes


def test_init():
    assert SentenceTransformerRerank()


def test_device():
    device = infer_torch_device() or "cpu"
    assert SentenceTransformerRerank()._device == device
