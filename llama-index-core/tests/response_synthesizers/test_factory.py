from llama_index.core import Settings
from llama_index.core.indices.prompt_helper import PromptHelper, ChatPromptHelper
from llama_index.core.llms.mock import MockLLM
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.base.llms.types import LLMMetadata
import pytest


@pytest.fixture(autouse=True)
def reset_settings():
    # Store old state
    old_llm = Settings._llm
    old_embed_model = Settings._embed_model
    old_callback_manager = Settings._callback_manager
    old_tokenizer = Settings._tokenizer
    old_node_parser = Settings._node_parser
    old_prompt_helper = Settings._prompt_helper
    old_chat_prompt_helper = Settings._chat_prompt_helper
    old_transformations = Settings._transformations

    # Reset state
    Settings._llm = None
    Settings._embed_model = None
    Settings._callback_manager = None
    Settings._tokenizer = None
    Settings._node_parser = None
    Settings._prompt_helper = None
    Settings._chat_prompt_helper = None
    Settings._transformations = None

    yield

    # Restore old state
    Settings._llm = old_llm
    Settings._embed_model = old_embed_model
    Settings._callback_manager = old_callback_manager
    Settings._tokenizer = old_tokenizer
    Settings._node_parser = old_node_parser
    Settings._prompt_helper = old_prompt_helper
    Settings._chat_prompt_helper = old_chat_prompt_helper
    Settings._transformations = old_transformations


class CustomMockLLM(MockLLM):
    custom_metadata: LLMMetadata = LLMMetadata(
        context_window=12345,
        num_output=543,
        is_chat_model=True,
    )

    @property
    def metadata(self) -> LLMMetadata:
        return self.custom_metadata


def test_prompt_helper_from_supplied_llm_metadata():
    # When no explicit prompt_helper is passed, and Settings.prompt_helper is NOT configured,
    # but a custom LLM is passed to get_response_synthesizer, it should generate the prompt helper
    # from the supplied LLM's metadata.
    custom_llm = CustomMockLLM()

    synthesizer = get_response_synthesizer(llm=custom_llm)
    # The synthesizer should have prompt_helper and chat_prompt_helper generated from custom_llm's metadata
    assert synthesizer._prompt_helper.context_window == 12345
    assert synthesizer._prompt_helper.num_output == 543
    assert synthesizer._chat_prompt_helper.context_window == 12345
    assert synthesizer._chat_prompt_helper.num_output == 543


def test_explicit_prompt_helper():
    # If prompt_helper is explicitly passed, it should be used.
    custom_prompt_helper = PromptHelper(context_window=9999, num_output=99)
    custom_chat_prompt_helper = ChatPromptHelper(context_window=9999, num_output=99)
    synthesizer = get_response_synthesizer(
        prompt_helper=custom_prompt_helper, chat_prompt_helper=custom_chat_prompt_helper
    )
    assert synthesizer._prompt_helper is custom_prompt_helper
    assert synthesizer._chat_prompt_helper is custom_chat_prompt_helper


def test_prompt_helper_from_global_settings():
    # If Settings.prompt_helper is explicitly configured, it should be used.
    global_prompt_helper = PromptHelper(context_window=8888, num_output=88)
    global_chat_prompt_helper = ChatPromptHelper(context_window=8888, num_output=88)
    Settings.prompt_helper = global_prompt_helper
    Settings.chat_prompt_helper = global_chat_prompt_helper

    synthesizer = get_response_synthesizer()
    assert synthesizer._prompt_helper is global_prompt_helper
    assert synthesizer._chat_prompt_helper is global_chat_prompt_helper


def test_prompt_helper_from_supplied_llm_metadata_direct_instantiation():
    # Verify direct instantiation of response synthesizers (bypassing the factory)
    # also falls back correctly to the supplied LLM's metadata.
    from llama_index.core.response_synthesizers.compact_and_refine import (
        CompactAndRefine,
    )

    custom_llm = CustomMockLLM()
    synthesizer = CompactAndRefine(llm=custom_llm)
    assert synthesizer._prompt_helper.context_window == 12345
    assert synthesizer._prompt_helper.num_output == 543
    assert synthesizer._chat_prompt_helper.context_window == 12345
    assert synthesizer._chat_prompt_helper.num_output == 543


def test_default_behavior_when_no_custom_llm_or_prompt_helper_supplied():
    # If no custom LLM or prompt helper is supplied, it falls back to the default LLM from Settings (which uses DEFAULT_CONTEXT_WINDOW)
    from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW

    synthesizer = get_response_synthesizer()
    # The default prompt helper should use DEFAULT_CONTEXT_WINDOW
    assert synthesizer._prompt_helper.context_window == DEFAULT_CONTEXT_WINDOW
    assert synthesizer._chat_prompt_helper.context_window == DEFAULT_CONTEXT_WINDOW


def test_prompt_helper_preserves_behavior_when_settings_llm_is_configured():
    # If Settings.llm is explicitly configured, it should be used to build the prompt helper
    custom_llm = CustomMockLLM()
    Settings.llm = custom_llm

    synthesizer = get_response_synthesizer()
    assert synthesizer._prompt_helper.context_window == 12345
    assert synthesizer._prompt_helper.num_output == 543
    assert synthesizer._chat_prompt_helper.context_window == 12345
    assert synthesizer._chat_prompt_helper.num_output == 543
