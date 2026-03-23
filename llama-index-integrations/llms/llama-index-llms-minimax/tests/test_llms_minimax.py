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


def test_default_model_is_m27():
    """Default model should be MiniMax-M2.7."""
    llm = MiniMax(api_key="test-key")
    assert llm.model == "MiniMax-M2.7"


def test_model_list_contains_m27():
    """Both M2.7 models should be in the context window map."""
    assert "MiniMax-M2.7" in MINIMAX_MODEL_TO_CONTEXT_WINDOW
    assert "MiniMax-M2.7-highspeed" in MINIMAX_MODEL_TO_CONTEXT_WINDOW


def test_m27_in_function_calling_models():
    """Both M2.7 models should support function calling."""
    assert "MiniMax-M2.7" in FUNCTION_CALLING_MODELS
    assert "MiniMax-M2.7-highspeed" in FUNCTION_CALLING_MODELS


def test_m27_appears_before_m25():
    """M2.7 models should appear before M2.5 in the model list."""
    keys = list(MINIMAX_MODEL_TO_CONTEXT_WINDOW.keys())
    m27_idx = keys.index("MiniMax-M2.7")
    m25_idx = keys.index("MiniMax-M2.5")
    assert m27_idx < m25_idx


def test_legacy_models_still_available():
    """Old M2.5 models should still be present."""
    assert "MiniMax-M2.5" in MINIMAX_MODEL_TO_CONTEXT_WINDOW
    assert "MiniMax-M2.5-highspeed" in MINIMAX_MODEL_TO_CONTEXT_WINDOW
    assert "MiniMax-M2.5" in FUNCTION_CALLING_MODELS
    assert "MiniMax-M2.5-highspeed" in FUNCTION_CALLING_MODELS


def test_context_window_for_m27():
    """M2.7 models should return the correct context window."""
    assert get_context_window("MiniMax-M2.7") == 204800
    assert get_context_window("MiniMax-M2.7-highspeed") == 204800
