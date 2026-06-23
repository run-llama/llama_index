import json
from unittest.mock import MagicMock, patch

import boto3
import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding, EmbeddingResponse
from llama_index.embeddings.bedrock import BedrockEmbedding


def test_class():
    names_of_base_classes = [b.__name__ for b in BedrockEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


def test_model_param_raises_error():
    """Test that passing 'model' instead of 'model_name' raises ValueError."""
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    with pytest.raises(ValueError, match="Please use 'model_name' instead"):
        BedrockEmbedding(model="cohere.embed-multilingual-v3", client=bedrock_client)


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


@pytest.mark.parametrize(
    ("model_name", "expected_provider"),
    [
        (
            "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1",
            "amazon",
        ),
        (
            "arn:aws:bedrock:us-east-1::foundation-model/us.amazon.titan-embed-text-v1",
            "amazon",
        ),
        (
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.amazon.titan-embed-text-v2:0",
            "amazon",
        ),
        (
            "arn:aws:bedrock:us-west-2::foundation-model/cohere.embed-english-v3",
            "cohere",
        ),
        ("anthropic.claude-v2", "anthropic"),
    ],
)
def test_get_provider(model_name, expected_provider):
    """Test _get_provider with various model name formats."""
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    embedding = BedrockEmbedding(
        model_name=model_name,
        client=bedrock_client,
    )
    assert embedding._get_provider() == expected_provider


def _make_invoke_model_response(body_dict: dict, input_token_count: int = 10) -> dict:
    """Build a mock boto3 invoke_model response with headers and body."""
    body_bytes = json.dumps(body_dict).encode("utf-8")
    stream = MagicMock()
    stream.read.return_value = body_bytes
    return {
        "body": stream,
        "ResponseMetadata": {
            "HTTPHeaders": {
                "x-amzn-bedrock-input-token-count": str(input_token_count),
            },
        },
    }


def test_get_embedding_extracts_token_count_from_header() -> None:
    """Token count is extracted from x-amzn-bedrock-input-token-count header."""
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    emb = BedrockEmbedding(
        model_name="amazon.titan-embed-text-v1",
        client=bedrock_client,
    )

    resp = _make_invoke_model_response(
        {"embedding": [0.1, 0.2, 0.3], "inputTextTokenCount": 5},
        input_token_count=5,
    )
    with patch.object(emb._client, "invoke_model", return_value=resp):
        result = emb._get_embedding("hello", "text")

    assert isinstance(result, EmbeddingResponse)
    assert result.embedding == [0.1, 0.2, 0.3]
    assert result.token_count == 5


def test_get_embedding_falls_back_to_body_token_count() -> None:
    """When header is missing, falls back to inputTextTokenCount in body."""
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    emb = BedrockEmbedding(
        model_name="amazon.titan-embed-text-v1",
        client=bedrock_client,
    )

    body_bytes = json.dumps({"embedding": [0.1, 0.2], "inputTextTokenCount": 7}).encode(
        "utf-8"
    )
    stream = MagicMock()
    stream.read.return_value = body_bytes
    resp = {
        "body": stream,
        "ResponseMetadata": {"HTTPHeaders": {}},
    }
    with patch.object(emb._client, "invoke_model", return_value=resp):
        result = emb._get_embedding("hello", "text")

    assert isinstance(result, EmbeddingResponse)
    assert result.token_count == 7
