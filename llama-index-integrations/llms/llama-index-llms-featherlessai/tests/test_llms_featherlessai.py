from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.featherlessai import FeatherlessLLM


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in FeatherlessLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_function_calling_enabled_by_default():
    llm = FeatherlessLLM(model="Qwen/Qwen3.8-27B", api_key="fake")
    assert llm.metadata.is_function_calling_model is True


def test_function_calling_can_be_disabled():
    llm = FeatherlessLLM(
        model="Qwen/Qwen3.8-27B", api_key="fake", is_function_calling_model=False
    )
    assert llm.metadata.is_function_calling_model is False


def test_context_window_override():
    llm = FeatherlessLLM(model="Qwen/Qwen3.8-27B", api_key="fake", context_window=32768)
    assert llm.metadata.context_window == 32768
