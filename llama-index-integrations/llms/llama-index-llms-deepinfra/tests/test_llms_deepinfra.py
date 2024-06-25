from llama_index.llms.deepinfra import DeepInfraLLM
from llama_index.core.llms.llm import LLM


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in DeepInfraLLM.__mro__]
    assert LLM.__name__ in names_of_base_classes


def test_deepinfra_llm_class():
    model = DeepInfraLLM(api_key="test")
    assert isinstance(model, LLM)


def test_deepinfra_llm_from_env():
    import os

    os.environ["DEEPINFRA_API_TOKEN"] = "test"
    model = DeepInfraLLM()
    assert isinstance(model, LLM)
