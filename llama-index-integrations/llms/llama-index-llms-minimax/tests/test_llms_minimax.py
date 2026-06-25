from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.minimax import MiniMax
from llama_index.llms.minimax.utils import (
    MINIMAX_MODEL_TO_CONTEXT_WINDOW,
    FUNCTION_CALLING_MODELS,
    get_context_window,
)


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in MiniMax.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_default_model_is_m3():
    """Default model should be MiniMax-M3."""
    llm = MiniMax(api_key="test-key")
    assert llm.model == "MiniMax-M3"


def test_model_list_contains_m3():
    """M3 should be in the context window map."""
    assert "MiniMax-M3" in MINIMAX_MODEL_TO_CONTEXT_WINDOW


def test_m3_in_function_calling_models():
    """M3 should support function calling."""
    assert "MiniMax-M3" in FUNCTION_CALLING_MODELS


def test_m3_appears_before_m27():
    """M3 should appear before M2.7 in the model list."""
    keys = list(MINIMAX_MODEL_TO_CONTEXT_WINDOW.keys())
    m3_idx = keys.index("MiniMax-M3")
    m27_idx = keys.index("MiniMax-M2.7")
    assert m3_idx < m27_idx


def test_m27_legacy_still_available():
    """M2.7 models should remain available for backward compatibility."""
    assert "MiniMax-M2.7" in MINIMAX_MODEL_TO_CONTEXT_WINDOW
    assert "MiniMax-M2.7-highspeed" in MINIMAX_MODEL_TO_CONTEXT_WINDOW
    assert "MiniMax-M2.7" in FUNCTION_CALLING_MODELS
    assert "MiniMax-M2.7-highspeed" in FUNCTION_CALLING_MODELS


def test_older_models_removed():
    """Older M2.5 models should no longer be present."""
    assert "MiniMax-M2.5" not in MINIMAX_MODEL_TO_CONTEXT_WINDOW
    assert "MiniMax-M2.5-highspeed" not in MINIMAX_MODEL_TO_CONTEXT_WINDOW
    assert "MiniMax-M2.5" not in FUNCTION_CALLING_MODELS
    assert "MiniMax-M2.5-highspeed" not in FUNCTION_CALLING_MODELS


def test_context_window_for_m3():
    """M3 should return the 524,288 context window."""
    assert get_context_window("MiniMax-M3") == 524288


def test_context_window_for_m27():
    """M2.7 models should return the correct context window."""
    assert get_context_window("MiniMax-M2.7") == 204800
    assert get_context_window("MiniMax-M2.7-highspeed") == 204800
