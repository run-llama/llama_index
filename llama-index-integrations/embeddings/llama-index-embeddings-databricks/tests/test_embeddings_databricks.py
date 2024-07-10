import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding

# TODO: Pants, for whatever reason, always breaks on tests that depend on openai embeddings
# It fails to resolve dependencies properly during testing, so this import will fail in CICD
try:
    from llama_index.embeddings.databricks import DatabricksEmbedding
except ImportError:
    DatabricksEmbedding = None


@pytest.mark.skipif(
    DatabricksEmbedding is None, reason="DatabricksEmbedding is missing"
)
def test_dashscope_embedding_class():
    names_of_base_classes = [b.__name__ for b in DatabricksEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes
