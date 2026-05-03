from pydantic import BaseModel

from llama_index.llms.ollama.base import get_additional_kwargs


def test_get_additional_kwargs():
    response = {"key1": "value1", "key2": "value2", "exclude_me": "value3"}
    exclude = ("exclude_me", "exclude_me_too")

    expected = {"key1": "value1", "key2": "value2"}

    actual = get_additional_kwargs(response, exclude)

    assert actual == expected


def test_get_additional_kwargs_pydantic_model():
    """Test that get_additional_kwargs handles Pydantic models (e.g. GenerateResponse
    from ollama>=0.4) in addition to plain dicts."""

    class FakeGenerateResponse(BaseModel):
        key1: str = "value1"
        key2: str = "value2"
        exclude_me: str = "value3"

    response = FakeGenerateResponse()
    exclude = ("exclude_me", "exclude_me_too")

    expected = {"key1": "value1", "key2": "value2"}

    actual = get_additional_kwargs(response, exclude)

    assert actual == expected
