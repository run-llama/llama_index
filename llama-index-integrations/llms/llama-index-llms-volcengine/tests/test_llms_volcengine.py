from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.volcengine import Volcengine
from llama_index.llms.volcengine.utils import (
    FUNCTION_CALLING_MODELS,
    VOLCENGINE_MODEL_TO_CONTEXT_WINDOW,
    get_context_window,
)


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in Volcengine.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes
    assert "OpenAILike" in names_of_base_classes


def test_class_name():
    assert Volcengine.class_name() == "Volcengine"


def test_context_window():
    assert (
        get_context_window("doubao-seed-2-1-pro-260628")
        == VOLCENGINE_MODEL_TO_CONTEXT_WINDOW["doubao-seed-2-1-pro-260628"]
    )
    assert (
        get_context_window("doubao-seed-evolving")
        == VOLCENGINE_MODEL_TO_CONTEXT_WINDOW["doubao-seed-evolving"]
    )
    assert get_context_window("unknown-model") == 256000


def test_function_calling_models():
    assert "doubao-seed-2-1-pro-260628" in FUNCTION_CALLING_MODELS
    assert "doubao-seed-evolving" in FUNCTION_CALLING_MODELS
    assert "deepseek-v4-pro-260425" in FUNCTION_CALLING_MODELS
