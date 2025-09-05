from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.cometapi import CometAPI


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in CometAPI.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_cometapi_initialization():
    llm = CometAPI(model="gpt-4o-mini", api_key="test_key")
    assert llm.model == "gpt-4o-mini"
    assert llm.api_key == "test_key"
