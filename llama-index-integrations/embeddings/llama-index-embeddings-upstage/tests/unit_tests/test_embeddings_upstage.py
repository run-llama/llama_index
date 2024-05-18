import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding

# skip tests if below import is not found
try:
    from llama_index.embeddings.upstage import UpstageEmbedding
except ImportError:
    pytest.skip("Cannot import UpstageEmbedding", allow_module_level=True)


def test_upstage_embedding_class():
    names_of_base_classes = [b.__name__ for b in UpstageEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


def upstage_embedding_fail_wrong_model():
    with pytest.raises(ValueError):
        UpstageEmbedding(model="foo")
