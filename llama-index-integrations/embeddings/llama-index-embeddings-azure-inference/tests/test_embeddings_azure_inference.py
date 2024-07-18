import os
import pytest
from llama_index.embeddings.azure_inference import AzureAIEmbeddingsModel
from llama_index.core.schema import TextNode


@pytest.mark.skipif(
    not {
        "AZURE_INFERENCE_ENDPOINT",
        "AZURE_INFERENCE_CREDENTIAL",
    }.issubset(set(os.environ)),
    reason="Azure AI endpoint and/or credential are not set.",
)
def test_embed():
    """Test the basic embedding functionality."""
    # In case the endpoint being tested serves more than one model
    model_name = os.environ.get("AZURE_INFERENCE_MODEL", None)

    embed_model = AzureAIEmbeddingsModel(model_name=model_name)

    nodes = [
        TextNode(
            text="Before college the two main things I worked on, "
            "outside of school, were writing and programming."
        )
    ]
    response = embed_model(nodes=nodes)

    assert len(response) == len(nodes)
    assert response[0].embedding
