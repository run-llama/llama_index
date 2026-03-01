# --- Tests for structured output support ---

from pydantic import BaseModel
from llama_index.llms.bedrock_converse.utils import (
    BEDROCK_STRUCTURED_OUTPUT_SUPPORTED_MODELS,
    is_bedrock_structured_output_supported,
)


class ExampleOutput(BaseModel):
    """Test output model for structured prediction."""

    name: str
    value: int


def test_is_bedrock_structured_output_supported_with_supported_models():
    """Test that supported models are correctly identified."""
    # Test direct model names
    assert is_bedrock_structured_output_supported(
        "anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    assert is_bedrock_structured_output_supported("amazon.nova-pro-v1:0")
    assert is_bedrock_structured_output_supported("deepseek.r1-v1:0")


def test_is_bedrock_structured_output_supported_with_region_prefix():
    """Test that region-prefixed models are correctly identified."""
    assert is_bedrock_structured_output_supported(
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    assert is_bedrock_structured_output_supported("eu.amazon.nova-pro-v1:0")
    assert is_bedrock_structured_output_supported("apac.deepseek.r1-v1:0")


def test_is_bedrock_structured_output_supported_with_unsupported_models():
    """Test that unsupported models return False."""
    assert not is_bedrock_structured_output_supported("anthropic.claude-v2")
    assert not is_bedrock_structured_output_supported("amazon.titan-text-lite-v1")
    assert not is_bedrock_structured_output_supported("ai21.j2-mid-v1")


def test_bedrock_structured_output_supported_models_list():
    """Test that the supported models list is populated."""
    assert len(BEDROCK_STRUCTURED_OUTPUT_SUPPORTED_MODELS) > 0
    assert (
        "anthropic.claude-haiku-4-5-20251001-v1:0"
        in BEDROCK_STRUCTURED_OUTPUT_SUPPORTED_MODELS
    )
    assert "amazon.nova-pro-v1:0" in BEDROCK_STRUCTURED_OUTPUT_SUPPORTED_MODELS
    assert "deepseek.r1-v1:0" in BEDROCK_STRUCTURED_OUTPUT_SUPPORTED_MODELS
