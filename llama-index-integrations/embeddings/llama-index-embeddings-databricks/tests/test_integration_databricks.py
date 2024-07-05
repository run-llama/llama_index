import os
import pytest

from llama_index.embeddings.databricks import DatabricksEmbedding


@pytest.mark.skipif(
    "DATABRICKS_API_KEY" not in os.environ or "DATABRICKS_API_BASE" not in os.environ,
    reason="DATABRICKS_API_KEY or DATABRICKS_API_BASE not set in environment",
)
def test_completion():
    embed_model = DatabricksEmbedding(model="databricks-bge-large-en")
    embeddings = embed_model.get_text_embedding(
        "The DatabricksEmbedding integration works great."
    )
    assert isinstance(embeddings, list)
    assert len(embeddings) == 1024
