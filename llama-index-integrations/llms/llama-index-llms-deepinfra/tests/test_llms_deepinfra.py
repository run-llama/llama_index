from llama_index.llms.deepinfra import DeepInfraLLM
from llama_index.core.base.llms.base import LLM
from unittest.mock import patch


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in DeepInfraLLM.__mro__]
    assert LLM.__name__ in names_of_base_classes


def test_deepinfra_llm_class():
    model = DeepInfraLLM()
    assert isinstance(model, LLM)


# Mocking the requests.post method


@patch("requests.post")
def test_complete(mock_post):
    model = DeepInfraLLM()
    mock_post.return_value.json.return_value = {"choices": [{"text": "Hello World!"}]}
    response = model.complete("Hello World!")
    assert response == "Hello World!"
