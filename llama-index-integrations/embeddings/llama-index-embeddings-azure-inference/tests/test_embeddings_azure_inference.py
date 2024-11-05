from unittest import mock

# import aiohttp to force Pants to include it in the required dependencies
import aiohttp  # noqa
import pytest
from azure.ai.inference.models import EmbeddingItem, EmbeddingsResult
from llama_index.core.schema import TextNode
from llama_index.embeddings.azure_inference import AzureAIEmbeddingsModel


@pytest.fixture()
def test_embed_model():
    with mock.patch(
        "llama_index.embeddings.azure_inference.base.EmbeddingsClient", autospec=True
    ):
        embed_model = AzureAIEmbeddingsModel(
            endpoint="https://my-endpoint.inference.ai.azure.com",
            credential="my-api-key",
            model_name="my_model_name",
        )
    embed_model._client.embed.return_value = EmbeddingsResult(
        data=[EmbeddingItem(embedding=[1.0, 2.0, 3.0], index=0)]
    )
    return embed_model


def test_embed(test_embed_model: AzureAIEmbeddingsModel):
    """Test the basic embedding functionality."""
    # In case the endpoint being tested serves more than one model
    nodes = [
        TextNode(
            text="Before college the two main things I worked on, "
            "outside of school, were writing and programming."
        )
    ]
    response = test_embed_model(nodes=nodes)

    assert len(response) == len(nodes)
    assert response[0].embedding


def test_get_metadata(test_embed_model: AzureAIEmbeddingsModel, caplog):
    """Tests if we can get model metadata back from the endpoint. If so,
    model_name should not be 'unknown'. Some endpoints may not support this
    and in those cases a warning should be logged.
    """
    assert (
        test_embed_model.model_name != "unknown"
        or "does not support model metadata retrieval" in caplog.text
    )
