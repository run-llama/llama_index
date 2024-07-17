import os
import pytest
from llama_index.embeddings.azure_inference import AzureAIEmbeddingsModel
from llama_index.core.schema import TextNode


@pytest.mark.skipif(
    not set(
        ["AZURE_INFERENCE_ENDPOINT_URL", "AZURE_INFERENCE_ENDPOINT_CREDENTIAL"]
    ).issubset(set(os.environ)),
    reason="Azure AI endpoint and/or credential are not set.",
)
def test_embed():
    """Test the basic embedding functionality."""
    embed_model = AzureAIEmbeddingsModel()

    nodes = [
        TextNode(
            text="Before college the two main things I worked on, "
            "outside of school, were writing and programming."
        )
    ]
    response = embed_model(nodes=nodes)

    assert len(response) == len(nodes)
    assert response[0].embedding
