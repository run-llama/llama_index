import boto3
import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.bedrock import BedrockEmbedding


def test_class():
    names_of_base_classes = [b.__name__ for b in BedrockEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


def test_get_provider_two_part_format():
    """Test _get_provider with 2-part model names (provider.model)."""
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

    embedding = BedrockEmbedding(
        model_name="amazon.titan-embed-text-v1", client=bedrock_client
    )
    assert embedding._get_provider() == "amazon"

    embedding = BedrockEmbedding(
        model_name="cohere.embed-english-v3", client=bedrock_client
    )
    assert embedding._get_provider() == "cohere"


def test_get_provider_three_part_format():
    """Test _get_provider with 3-part model names (region.provider.model)."""
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

    embedding = BedrockEmbedding(
        model_name="us.amazon.titan-embed-text-v1", client=bedrock_client
    )
    assert embedding._get_provider() == "amazon"

    embedding = BedrockEmbedding(
        model_name="eu.cohere.embed-english-v3", client=bedrock_client
    )
    assert embedding._get_provider() == "cohere"

    embedding = BedrockEmbedding(
        model_name="global.amazon.titan-embed-text-v2", client=bedrock_client
    )
    assert embedding._get_provider() == "amazon"


def test_get_provider_invalid_format():
    """Test _get_provider raises ValueError for invalid model name formats."""
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

    embedding = BedrockEmbedding(model_name="invalid", client=bedrock_client)
    with pytest.raises(ValueError, match="Unexpected number of parts in model_name"):
        embedding._get_provider()

    embedding = BedrockEmbedding(
        model_name="too.many.parts.in.name", client=bedrock_client
    )
    with pytest.raises(ValueError, match="Unexpected number of parts in model_name"):
        embedding._get_provider()
