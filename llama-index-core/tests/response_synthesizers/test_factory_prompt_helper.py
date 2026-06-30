"""Test prompt_helper initialization in get_response_synthesizer factory."""

from llama_index.core.llms import MockLLM
from llama_index.core.response_synthesizers.factory import get_response_synthesizer
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.settings import Settings


def test_prompt_helper_respects_llm_metadata():
    """Test that prompt_helper is initialized from LLM metadata when not explicitly set."""
    # Create an LLM with custom context window
    custom_context_window = 8192
    llm = MockLLM(max_tokens=custom_context_window)

    # Store original settings
    original_prompt_helper = Settings._prompt_helper
    original_llm = Settings._llm

    try:
        # Reset settings to ensure clean state
        Settings._prompt_helper = None
        Settings._llm = None

        # Get response synthesizer with custom LLM
        synthesizer = get_response_synthesizer(
            llm=llm,
            response_mode=ResponseMode.COMPACT,
        )

        # Verify that prompt_helper uses LLM's context window, not default 3900
        assert synthesizer._prompt_helper.context_window == custom_context_window
        assert synthesizer._prompt_helper.context_window != 3900

    finally:
        # Restore original settings
        Settings._prompt_helper = original_prompt_helper
        Settings._llm = original_llm


def test_prompt_helper_explicit_setting_takes_precedence():
    """Test that explicit Settings._prompt_helper takes precedence over LLM metadata."""
    from llama_index.core.indices.prompt_helper import PromptHelper

    # Create an LLM with one context window
    llm = MockLLM(max_tokens=8192)

    # Create a prompt_helper with different context window
    explicit_context_window = 16384
    explicit_prompt_helper = PromptHelper(context_window=explicit_context_window)

    # Store original settings
    original_prompt_helper = Settings._prompt_helper
    original_llm = Settings._llm

    try:
        # Set explicit prompt_helper in settings
        Settings._prompt_helper = explicit_prompt_helper
        Settings._llm = None

        # Get response synthesizer with custom LLM
        synthesizer = get_response_synthesizer(
            llm=llm,
            response_mode=ResponseMode.COMPACT,
        )

        # Verify that explicit Settings._prompt_helper takes precedence
        assert synthesizer._prompt_helper.context_window == explicit_context_window
        assert synthesizer._prompt_helper.context_window != 8192

    finally:
        # Restore original settings
        Settings._prompt_helper = original_prompt_helper
        Settings._llm = original_llm
