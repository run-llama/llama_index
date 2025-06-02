import pytest

from llama_index.llms.bedrock.utils import (
    get_provider,
    MetaProvider,
    MistralProvider,
    AmazonProvider,
    AnthropicProvider,
    CohereProvider,
    Ai21Provider,
)


@pytest.mark.parametrize(
    ("model", "provider_cls"),
    [
        ("meta.llama3-3-70b-instruct-v1:0", MetaProvider),
        (
            "arn:aws:bedrock:us-east-2::foundation-model/meta.llama3-3-70b-instruct-v1:0",
            MetaProvider,
        ),
        (
            "arn:aws:bedrock:eu-central-1:143111710283:inference-profile/eu.meta.llama3-2-1b-instruct-v1:0",
            MetaProvider,
        ),
        ("amazon.titan-text-express-v1", AmazonProvider),
        (
            "arn:aws:bedrock:eu-central-1:143111710283:inference-profile/eu.amazon.nova-lite-v1:0",
            AmazonProvider,
        ),
        ("anthropic.claude-3-5-sonnet-20240620-v1:0", AnthropicProvider),
        (
            "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-3-7-sonnet-20250219-v1:0",
            AnthropicProvider,
        ),
        ("mistral.mistral-large-2402", MistralProvider),
        (
            "arn:aws:bedrock:eu-west-2::foundation-model/mistral.mistral-large-2402-v1:0",
            MistralProvider,
        ),
        ("cohere.command-text-v14", CohereProvider),
        (
            "arn:aws:bedrock:us-east-1::foundation-model/cohere.command-text-v14",
            CohereProvider,
        ),
        ("ai21.jamba-1-5-large-v1:0", Ai21Provider),
        (
            "arn:aws:bedrock:us-east-1::foundation-model/ai21.jamba-1-5-large-v1:0",
            Ai21Provider,
        ),
    ],
)
def test_get_provider(model: str, provider_cls):
    """Test that get_provider returns the expected provider class."""
    assert isinstance(get_provider(model=model), provider_cls)
