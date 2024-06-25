from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.upstage import Upstage


def test_text_inference_llm_class():
    names_of_base_classes = [b.__name__ for b in Upstage.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_upstage_llm_api_key_alias():
    api_key = "test_key"
    embedding1 = Upstage(api_key=api_key)
    embedding2 = Upstage(upstage_api_key=api_key)
    embedding3 = Upstage(error_api_key=api_key)

    assert embedding1.api_key == api_key
    assert embedding2.api_key == api_key
    assert embedding3.api_key == ""
