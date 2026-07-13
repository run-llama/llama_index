from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.crusoe import Crusoe
from llama_index.llms.crusoe.utils import (
    DEFAULT_CRUSOE_API_BASE,
    crusoe_modelname_to_contextsize,
    is_function_calling_model,
)


def test_crusoe_class_hierarchy():
    names_of_base_classes = [b.__name__ for b in Crusoe.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_crusoe_default_model():
    llm = Crusoe(api_key="fake-key")
    assert llm.model == "zai/GLM-5.2"


def test_crusoe_default_api_base():
    llm = Crusoe(api_key="fake-key")
    assert llm.api_base == DEFAULT_CRUSOE_API_BASE


def test_crusoe_class_name():
    assert Crusoe.class_name() == "Crusoe_LLM"


def test_crusoe_available_models():
    models = Crusoe.available_models()
    assert "zai/GLM-5.2" in models
    assert "google/gemma-4-31b-it" in models


def test_crusoe_metadata_known_model():
    llm = Crusoe(model="zai/GLM-5.2", api_key="fake-key")
    assert llm.metadata.context_window == 262144
    assert llm.metadata.is_chat_model is True
    assert llm.metadata.model_name == "zai/GLM-5.2"


def test_crusoe_metadata_custom_context_window():
    llm = Crusoe(api_key="fake-key", context_window=65536)
    assert llm.metadata.context_window == 65536


def test_crusoe_metadata_unknown_model_fallback():
    llm = Crusoe(model="some/future-model", api_key="fake-key")
    assert llm.metadata.context_window == 262144


def test_crusoe_function_calling_model():
    llm = Crusoe(model="zai/GLM-5.2", api_key="fake-key")
    assert llm.metadata.is_function_calling_model is True


def test_crusoe_non_function_calling_model():
    llm = Crusoe(model="google/gemma-4-31b-it", api_key="fake-key")
    assert llm.metadata.is_function_calling_model is False


def test_crusoe_custom_is_function_calling():
    llm = Crusoe(
        model="google/gemma-4-31b-it",
        api_key="fake-key",
        is_function_calling=True,
    )
    assert llm.metadata.is_function_calling_model is True


def test_crusoe_modelname_to_contextsize():
    assert crusoe_modelname_to_contextsize("zai/GLM-5.2") == 262144
    assert crusoe_modelname_to_contextsize("google/gemma-4-31b-it") == 262144


def test_is_function_calling_model():
    assert is_function_calling_model("zai/GLM-5.2") is True
    assert is_function_calling_model("google/gemma-4-31b-it") is False
