import pytest

from llama_index.llms.anthropic.utils import (
    is_anthropic_prompt_caching_supported_model,
    ANTHROPIC_PROMPT_CACHING_SUPPORTED_MODELS,
    update_tool_calls,
    is_anthropic_structured_output_supported,
    STRUCTURED_OUTPUT_SUPPORT,
    messages_to_anthropic_beta_messages,
    blocks_to_anthropic_blocks,
    blocks_to_anthropic_beta_blocks,
    messages_to_anthropic_messages,
)
from llama_index.core.base.llms.types import (
    ToolCallBlock,
    TextBlock,
    ChatMessage,
    ThinkingBlock,
    ImageBlock,
    DocumentBlock,
    MessageRole,
)
from anthropic.types.beta import (
    BetaTextBlockParam,
    BetaThinkingBlockParam,
    BetaToolUseBlockParam,
    BetaToolResultBlockParam,
)


class TestAnthropicPromptCachingSupport:
    """Test suite for Anthropic prompt caching model validation."""

    def test_claude_4_5_opus_supported(self):
        """Test Claude 4.5 Opus models support prompt caching."""
        assert is_anthropic_prompt_caching_supported_model("claude-opus-4-5-20251101")
        assert is_anthropic_prompt_caching_supported_model("claude-opus-4-5")

    def test_claude_4_1_opus_supported(self):
        """Test Claude 4.1 Opus models support prompt caching."""
        assert is_anthropic_prompt_caching_supported_model("claude-opus-4-1-20250805")
        assert is_anthropic_prompt_caching_supported_model("claude-opus-4-1")

    def test_claude_4_opus_supported(self):
        """Test Claude 4 Opus models support prompt caching."""
        assert is_anthropic_prompt_caching_supported_model("claude-opus-4-20250514")
        assert is_anthropic_prompt_caching_supported_model("claude-opus-4-0")
        assert is_anthropic_prompt_caching_supported_model("claude-4-opus-20250514")

    def test_claude_4_5_sonnet_supported(self):
        """Test Claude 4.5 Sonnet models support prompt caching."""
        assert is_anthropic_prompt_caching_supported_model("claude-sonnet-4-5-20250929")
        assert is_anthropic_prompt_caching_supported_model("claude-sonnet-4-5")

    def test_claude_4_sonnet_supported(self):
        """Test Claude 4 Sonnet models support prompt caching."""
        assert is_anthropic_prompt_caching_supported_model("claude-sonnet-4-20250514")
        assert is_anthropic_prompt_caching_supported_model("claude-sonnet-4-0")
        assert is_anthropic_prompt_caching_supported_model("claude-4-sonnet-20250514")

    def test_claude_3_7_sonnet_supported(self):
        """Test Claude 3.7 Sonnet models support prompt caching."""
        assert is_anthropic_prompt_caching_supported_model("claude-3-7-sonnet-20250219")
        assert is_anthropic_prompt_caching_supported_model("claude-3-7-sonnet-latest")

    def test_claude_3_5_sonnet_supported(self):
        """Test Claude 3.5 Sonnet models support prompt caching."""
        assert is_anthropic_prompt_caching_supported_model("claude-3-5-sonnet-20241022")
        assert is_anthropic_prompt_caching_supported_model("claude-3-5-sonnet-20240620")
        assert is_anthropic_prompt_caching_supported_model("claude-3-5-sonnet-latest")

    def test_claude_4_5_haiku_supported(self):
        """Test Claude 4.5 Haiku models support prompt caching."""
        assert is_anthropic_prompt_caching_supported_model("claude-haiku-4-5-20251001")
        assert is_anthropic_prompt_caching_supported_model("claude-haiku-4-5")

    def test_claude_3_5_haiku_supported(self):
        """Test Claude 3.5 Haiku models support prompt caching."""
        assert is_anthropic_prompt_caching_supported_model("claude-3-5-haiku-20241022")
        assert is_anthropic_prompt_caching_supported_model("claude-3-5-haiku-latest")

    def test_claude_3_haiku_supported(self):
        """Test Claude 3 Haiku models support prompt caching."""
        assert is_anthropic_prompt_caching_supported_model("claude-3-haiku-20240307")
        assert is_anthropic_prompt_caching_supported_model("claude-3-haiku-latest")

    def test_claude_3_opus_deprecated_but_supported(self):
        """Test deprecated Claude 3 Opus models still support prompt caching."""
        assert is_anthropic_prompt_caching_supported_model("claude-3-opus-20240229")
        assert is_anthropic_prompt_caching_supported_model("claude-3-opus-latest")

    def test_claude_2_not_supported(self):
        """Test Claude 2.x models do not support prompt caching."""
        assert not is_anthropic_prompt_caching_supported_model("claude-2")
        assert not is_anthropic_prompt_caching_supported_model("claude-2.0")
        assert not is_anthropic_prompt_caching_supported_model("claude-2.1")

    def test_claude_instant_not_supported(self):
        """Test Claude Instant models do not support prompt caching."""
        assert not is_anthropic_prompt_caching_supported_model("claude-instant-1")
        assert not is_anthropic_prompt_caching_supported_model("claude-instant-1.2")

    def test_invalid_model_not_supported(self):
        """Test invalid or unknown model names return False."""
        assert not is_anthropic_prompt_caching_supported_model("invalid-model")
        assert not is_anthropic_prompt_caching_supported_model("")
        assert not is_anthropic_prompt_caching_supported_model("gpt-4")
        assert not is_anthropic_prompt_caching_supported_model("claude-nonexistent")

    def test_constant_contains_all_supported_models(self):
        """Test that the constant tuple contains expected model patterns."""
        assert len(ANTHROPIC_PROMPT_CACHING_SUPPORTED_MODELS) > 0

        expected_patterns = [
            "claude-opus-4-5",
            "claude-opus-4-1",
            "claude-opus-4-0",
            "claude-sonnet-4-5",
            "claude-sonnet-4-0",
            "claude-3-7-sonnet",
            "claude-3-5-sonnet",
            "claude-haiku-4-5",
            "claude-3-5-haiku",
            "claude-3-haiku",
            "claude-3-opus",
        ]

        for pattern in expected_patterns:
            has_pattern = any(
                pattern in model for model in ANTHROPIC_PROMPT_CACHING_SUPPORTED_MODELS
            )
            assert has_pattern, (
                f"Expected pattern '{pattern}' not found in supported models"
            )

    def test_case_sensitivity(self):
        """Test that model name matching is case-sensitive."""
        assert is_anthropic_prompt_caching_supported_model("claude-sonnet-4-5-20250929")
        assert not is_anthropic_prompt_caching_supported_model(
            "Claude-Sonnet-4-5-20250929"
        )
        assert not is_anthropic_prompt_caching_supported_model(
            "CLAUDE-SONNET-4-5-20250929"
        )


def test_update_tool_calls() -> None:
    blocks = [TextBlock(text="hello world")]
    update_tool_calls(
        blocks, ToolCallBlock(tool_call_id="1", tool_name="hello", tool_kwargs={})
    )  # type: ignore
    assert len(blocks) == 2
    assert isinstance(blocks[1], ToolCallBlock)
    assert blocks[1].tool_call_id == "1"
    assert blocks[1].tool_name == "hello"
    assert blocks[1].tool_kwargs == {}
    update_tool_calls(
        blocks,
        ToolCallBlock(
            tool_call_id="1", tool_name="hello", tool_kwargs={"name": "John"}
        ),
    )  # type: ignore
    assert len(blocks) == 2
    assert isinstance(blocks[1], ToolCallBlock)
    assert blocks[1].tool_call_id == "1"
    assert blocks[1].tool_name == "hello"
    assert blocks[1].tool_kwargs == {"name": "John"}
    update_tool_calls(
        blocks, ToolCallBlock(tool_call_id="2", tool_name="hello", tool_kwargs={})
    )  # type: ignore
    assert len(blocks) == 3
    assert isinstance(blocks[2], ToolCallBlock)
    assert blocks[2].tool_call_id == "2"
    assert blocks[2].tool_name == "hello"
    assert blocks[2].tool_kwargs == {}


def test_messages_to_anthropic_beta_messages() -> None:
    messages_legit = [
        ChatMessage(
            role="system",
            blocks=[
                TextBlock(text="this is a"),
                TextBlock(text="system message"),
                ThinkingBlock(
                    content="this will be ignored when building the system prompt",
                    additional_information={"signature": "ignored"},
                ),
            ],
        ),
        ChatMessage(
            role="user",
            blocks=[
                TextBlock(text="this is a test: say hello to World."),
            ],
        ),
        ChatMessage(
            blocks=[
                ThinkingBlock(
                    content="user asks me to say hello to World",
                    additional_information={"signature": "hello"},
                ),
                ToolCallBlock(
                    tool_name="say_hello",
                    tool_kwargs={"name": "World"},
                    tool_call_id="1",
                ),
            ],
            role="assistant",
        ),
        ChatMessage(
            blocks=[TextBlock(text="Hello World!")],
            role=MessageRole.TOOL.value,
            additional_kwargs={"tool_call_id": "1"},
        ),
    ]
    ant_messages, system = messages_to_anthropic_beta_messages(messages_legit)
    assert (
        len(ant_messages) == len(messages_legit) - 1
    )  # system message is not captured
    assert system == "this is a\nsystem message"
    assert ant_messages[0]["role"] == "user"
    assert ant_messages[0]["content"] == [
        BetaTextBlockParam(type="text", text="this is a test: say hello to World.")
    ]
    assert ant_messages[1]["role"] == "assistant"
    assert ant_messages[1]["content"] == [
        BetaThinkingBlockParam(
            type="thinking",
            thinking="user asks me to say hello to World",
            signature="hello",
        ),
        BetaToolUseBlockParam(
            name="say_hello", input={"name": "World"}, id="1", type="tool_use"
        ),
    ]
    assert ant_messages[2]["role"] == "user"
    assert ant_messages[2]["content"] == [
        BetaToolResultBlockParam(
            tool_use_id="1",
            content=[BetaTextBlockParam(type="text", text="Hello World!")],
            type="tool_result",
        )
    ]
    messages_image = [
        ChatMessage(
            role="user",
            blocks=[
                ImageBlock(
                    block_type="image", image=b"helloworld", image_mimetype="image/png"
                ),
            ],
        ),
    ]
    ant_messages, system = messages_to_anthropic_beta_messages(messages_image)
    assert not system
    assert len(ant_messages) == len(messages_image)
    assert ant_messages[0]["role"] == "user"
    assert isinstance(ant_messages[0]["content"], list)
    assert isinstance(ant_messages[0]["content"][0], dict)
    assert ant_messages[0]["content"][0]["type"] == "image"
    assert ant_messages[0]["content"][0]["source"]["type"] == "base64"
    assert ant_messages[0]["content"][0]["source"]["media_type"] == "image/png"
    messages_image_unsupp = [
        ChatMessage(
            role="user",
            blocks=[
                ImageBlock(
                    block_type="image", image=b"helloworld", image_mimetype="image/tiff"
                ),
            ],
        ),
    ]
    with pytest.raises(ValueError, match="Image mimetype image/tiff not supported"):
        messages_to_anthropic_beta_messages(messages_image_unsupp)
    messages_document_unsupp = [
        ChatMessage(
            role="user",
            blocks=[
                DocumentBlock(data=b"document", document_mimetype="application/pdf")
            ],
        )
    ]
    with pytest.raises(ValueError, match=f"Block type not supported: document"):
        messages_to_anthropic_beta_messages(messages_document_unsupp)


def test_is_anthropic_structured_output_supported() -> None:
    for model in STRUCTURED_OUTPUT_SUPPORT:
        assert is_anthropic_structured_output_supported(model)
    assert not is_anthropic_structured_output_supported("claude-sonnet-4-0")


class TestCacheControlOnlyLastBlock:
    """
    Test that cache_control is applied only to the last block.

    Fixes: https://github.com/run-llama/llama_index/issues/20854
    When a message has many blocks and cache_control is set, only the last
    block should receive cache_control to stay within Anthropic's limit of
    4 blocks with cache_control per request.
    """

    def test_blocks_to_anthropic_blocks_cache_control_only_on_last(self):
        """cache_control should only appear on the last block."""
        blocks = [
            TextBlock(text="block 1"),
            TextBlock(text="block 2"),
            TextBlock(text="block 3"),
            TextBlock(text="block 4"),
            TextBlock(text="block 5"),
        ]
        kwargs = {"cache_control": {"type": "ephemeral"}}
        result = blocks_to_anthropic_blocks(blocks, kwargs)

        assert len(result) == 5
        # Only the last block should have cache_control
        for i in range(4):
            assert "cache_control" not in result[i], (
                f"Block {i} should not have cache_control"
            )
        assert result[4]["cache_control"] == {"type": "ephemeral"}

    def test_blocks_to_anthropic_blocks_no_cache_control(self):
        """Without cache_control in kwargs, no block should have it."""
        blocks = [
            TextBlock(text="block 1"),
            TextBlock(text="block 2"),
        ]
        kwargs = {}
        result = blocks_to_anthropic_blocks(blocks, kwargs)

        assert len(result) == 2
        for block in result:
            assert "cache_control" not in block

    def test_blocks_to_anthropic_blocks_single_block_with_cache(self):
        """Single block should get cache_control when set."""
        blocks = [TextBlock(text="only block")]
        kwargs = {"cache_control": {"type": "ephemeral"}}
        result = blocks_to_anthropic_blocks(blocks, kwargs)

        assert len(result) == 1
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_blocks_to_anthropic_beta_blocks_cache_control_only_on_last(self):
        """Beta: cache_control should only appear on the last block."""
        blocks = [
            TextBlock(text="block 1"),
            TextBlock(text="block 2"),
            TextBlock(text="block 3"),
            TextBlock(text="block 4"),
            TextBlock(text="block 5"),
        ]
        kwargs = {"cache_control": {"type": "ephemeral"}}
        result = blocks_to_anthropic_beta_blocks(blocks, kwargs)

        assert len(result) == 5
        # Only the last block should have cache_control
        for i in range(4):
            assert "cache_control" not in result[i], (
                f"Block {i} should not have cache_control"
            )
        assert result[4]["cache_control"] == {"type": "ephemeral"}

    def test_blocks_to_anthropic_beta_blocks_no_cache_control(self):
        """Beta: without cache_control in kwargs, no block should have it."""
        blocks = [
            TextBlock(text="block 1"),
            TextBlock(text="block 2"),
        ]
        kwargs = {}
        result = blocks_to_anthropic_beta_blocks(blocks, kwargs)

        assert len(result) == 2
        for block in result:
            assert "cache_control" not in block

    def test_messages_to_anthropic_messages_cache_idx_last_block_only(self):
        """
        With cache_idx, cache_control should only be on the last block
        of each message, not on every block.
        """
        messages = [
            ChatMessage(
                role="user",
                blocks=[
                    TextBlock(text="first block"),
                    TextBlock(text="second block"),
                    TextBlock(text="third block"),
                ],
            ),
        ]
        ant_messages, _ = messages_to_anthropic_messages(messages, cache_idx=0)

        assert len(ant_messages) == 1
        content = ant_messages[0]["content"]
        assert len(content) == 3
        # Only the last block should have cache_control
        for i in range(2):
            assert "cache_control" not in content[i], (
                f"Block {i} should not have cache_control"
            )
        assert content[2]["cache_control"] == {"type": "ephemeral"}

    def test_many_blocks_stay_under_api_limit(self):
        """
        Simulates the exact scenario from issue #20854: 29 blocks with
        cache_control should not exceed Anthropic's 4-block limit.
        """
        blocks = [TextBlock(text=f"block {i}") for i in range(29)]
        kwargs = {"cache_control": {"type": "ephemeral"}}
        result = blocks_to_anthropic_blocks(blocks, kwargs)

        cache_control_count = sum(1 for block in result if "cache_control" in block)
        assert cache_control_count == 1, (
            f"Expected 1 block with cache_control, found {cache_control_count}"
        )
        # Verify it's on the last block
        assert "cache_control" in result[-1]
