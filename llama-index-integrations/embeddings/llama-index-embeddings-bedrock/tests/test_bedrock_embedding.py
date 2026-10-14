import json
import boto3
import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.bedrock import BedrockEmbedding, Models


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


def test_empty_payload_rejected_before_aws_call():
    """
    Reject an empty texts list locally instead of hitting AWS.

    Bedrock answers an empty ``texts`` payload with a misleading
    ``ValidationException: Invalid parameter combination`` (#22382).
    """
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    embedding = BedrockEmbedding(
        model_name="cohere.embed-english-v3", client=bedrock_client
    )

    with pytest.raises(ValueError, match="at least one text"):
        embedding._get_embedding([], "text")

    with pytest.raises(ValueError, match="at least one text"):
        embedding._get_request_body("cohere", [], "text")


def test_non_empty_payload_still_builds_request_body():
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    embedding = BedrockEmbedding(
        model_name="cohere.embed-english-v3", client=bedrock_client
    )

    body = embedding._get_request_body("cohere", ["hello", "world"], "text")
    assert '"hello"' in body and '"world"' in body

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


def test_cohere_additional_kwargs_output_dimension():
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    embedding = BedrockEmbedding(
        model_name=Models.COHERE_EMBED_V4,
        client=bedrock_client,
        additional_kwargs={"output_dimension": 256},
    )
    body = json.loads(embedding._get_request_body("cohere", ["hello"], "text"))
    assert body == {
        "output_dimension": 256,
        "texts": ["hello"],
        "input_type": "search_document",
    }


def test_cohere_additional_kwargs_truncate():
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    embedding = BedrockEmbedding(
        model_name=Models.COHERE_EMBED_V4,
        client=bedrock_client,
        additional_kwargs={"truncate": "RIGHT"},
    )
    body = json.loads(embedding._get_request_body("cohere", ["hello"], "text"))
    assert body == {
        "truncate": "RIGHT",
        "texts": ["hello"],
        "input_type": "search_document",
    }


def test_cohere_query_path_additional_kwargs():
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    embedding = BedrockEmbedding(
        model_name=Models.COHERE_EMBED_V4,
        client=bedrock_client,
        additional_kwargs={"output_dimension": 512},
    )
    body = json.loads(embedding._get_request_body("cohere", "test query", "query"))
    assert body == {
        "output_dimension": 512,
        "texts": ["test query"],
        "input_type": "search_query",
    }


def test_cohere_core_fields_authoritative():
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    embedding = BedrockEmbedding(
        model_name=Models.COHERE_EMBED_V4,
        client=bedrock_client,
        additional_kwargs={
            "texts": ["corrupt"],
            "input_type": "bad",
            "output_dimension": 1024,
        },
    )
    body = json.loads(embedding._get_request_body("cohere", ["legit"], "text"))
    assert body == {
        "output_dimension": 1024,
        "texts": ["legit"],
        "input_type": "search_document",
    }


def test_cohere_embedding_types_validation():
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    embedding = BedrockEmbedding(
        model_name=Models.COHERE_EMBED_V4,
        client=bedrock_client,
        additional_kwargs={"embedding_types": ["int8"]},
    )
    with pytest.raises(
        ValueError,
        match="BedrockEmbedding only supports 'float' embeddings",
    ):
        embedding._get_request_body("cohere", ["hello"], "text")


def test_cohere_embedding_types_string_rejected():
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    embedding = BedrockEmbedding(
        model_name=Models.COHERE_EMBED_V4,
        client=bedrock_client,
        additional_kwargs={"embedding_types": "float"},
    )
    with pytest.raises(
        ValueError,
        match="BedrockEmbedding only supports 'float' embeddings",
    ):
        embedding._get_request_body("cohere", ["hello"], "text")


def test_cohere_embedding_types_valid_float():
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    embedding = BedrockEmbedding(
        model_name=Models.COHERE_EMBED_V4,
        client=bedrock_client,
        additional_kwargs={"embedding_types": ["float", "int8"]},
    )
    body = json.loads(embedding._get_request_body("cohere", ["hello"], "text"))
    assert body == {
        "embedding_types": ["float", "int8"],
        "texts": ["hello"],
        "input_type": "search_document",
    }


def test_cohere_empty_kwargs_baseline():
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    embedding = BedrockEmbedding(
        model_name=Models.COHERE_EMBED_V4,
        client=bedrock_client,
    )
    body = json.loads(embedding._get_request_body("cohere", ["hello"], "text"))
    assert body == {
        "texts": ["hello"],
        "input_type": "search_document",
    }
