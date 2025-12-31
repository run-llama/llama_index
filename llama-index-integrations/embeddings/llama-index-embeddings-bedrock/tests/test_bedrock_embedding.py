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


def test_get_provider_with_foundation_arn():
    """Test _get_provider with Foundation Model ARNs."""
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

    # Standard ARN
    embedding = BedrockEmbedding(
        model_name="arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1",
        client=bedrock_client,
    )
    assert embedding._get_provider() == "amazon"

    # ARN with region prefix in model ID
    embedding = BedrockEmbedding(
        model_name="arn:aws:bedrock:us-east-1::foundation-model/us.amazon.titan-embed-text-v1",
        client=bedrock_client,
    )
    assert embedding._get_provider() == "amazon"


def test_get_provider_with_inference_profile_arn():
    """Test _get_provider with Inference Profile ARNs."""
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

    # Inference Profile ARN (often region prefixed)
    embedding = BedrockEmbedding(
        model_name="arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.amazon.titan-embed-text-v2:0",
        client=bedrock_client,
    )
    assert embedding._get_provider() == "amazon"
