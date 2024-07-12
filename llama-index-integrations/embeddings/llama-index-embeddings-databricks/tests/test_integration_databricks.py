import os
import pytest

# TODO: Pants, for whatever reason, always breaks on tests that depend on openai embeddings
# It fails to resolve dependencies properly during testing, so this import will fail in CICD
try:
    from llama_index.embeddings.databricks import DatabricksEmbedding
except ImportError:
    DatabricksEmbedding = None


@pytest.mark.skipif(
    "DATABRICKS_TOKEN" not in os.environ
    or "DATABRICKS_SERVING_ENDPOINT" not in os.environ
    or DatabricksEmbedding is None,
    reason="DATABRICKS_TOKEN or DATABRICKS_SERVING_ENDPOINT not set in environment",
)
def test_completion():
    embed_model = DatabricksEmbedding(model="databricks-bge-large-en")
    embeddings = embed_model.get_text_embedding(
        "The DatabricksEmbedding integration works great."
    )
    assert isinstance(embeddings, list)
    assert len(embeddings) == 1024
